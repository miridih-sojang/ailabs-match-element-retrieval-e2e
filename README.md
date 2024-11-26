# ailabs-miridih-trainer

## 1. 목적
- Deepspeed, Accelator에 Huggingface의 Trainer를 사용하면서 Custom Metric을 추가하는 방법을 설명하는 코드 
## 2. 개요
- Custom Metric을 추가하는데는 2가지 종류가 존재하고, 추가하는 법이 상이함.
1) DataLoader를 공유하는 경우
2) DataLoader를 공유 하지 않는 경우 ( 어울리는 요소 찾기 프로젝트 )

### 2-1) DataLoader를 공유하는 경우  
- `compute_metrics` 함수 정의와 Trainer 객체 생성 부분에 `compute_metrics=compute_metrics` 추가 필요

#### - compute_metrics 예시
- Input : `pred` 변수 내의 `predictions`, `label_ids`, `inputs` 변수로 값들 접근 가능 ( 모두 2-D Array Numpy)  
- Output : `{'Recall@1': 0.1, 'Recall@4': 0.2, 'Recall@5': 0.1, ..., 'MRR@100': 0.5}` ( Dict 형태의 값, 해당 값이 Wandb에 Key-Value로 Logging)

```python
def compute_metrics(pred):
    test_vocabulary_path = '/data/match_element_retrieval/cache/search_dataset/test'
    test_dataset_df = pd.read_csv(f"{test_vocabulary_path}/csv/test_vocabulary.csv")
    test_dataset_df = test_dataset_df[~test_dataset_df.candidate_resourceKey.isna()]
    test_dataset_df['x_image_path'] = test_dataset_df['x_image_path'].apply(
        lambda x: f"{test_vocabulary_path}/images/{x}.png")
    test_dataset_df[['candidate_resourceKey', 'x_info', 'y_info']] = test_dataset_df[
        ['candidate_resourceKey', 'x_info', 'y_info']].map(lambda x: eval(x))
    test_dataset_df = test_dataset_df[test_dataset_df.candidate_resourceKey.str.len() < 400]
    test_dataset_df = test_dataset_df[
        ['search_word', 'template_idx', 'page_num', 'x_image_path', 'resourceKey', 'candidate_resourceKey']]
    test_dataset_df['score'] = pred.predictions.tolist()
    test_dataset_df['score'] = test_dataset_df['score'].apply(
        lambda x: np.array([instance_x for instance_x in x if instance_x != -10]))
    return evaluate(test_dataset_df)

def evaluate(df, top_k_list=[1, 4, 5, 10, 20, 50, 100]):
    metric = {}
    df[['sorted_score', 'sorted_candidate_resourcekey']] = df.apply(sorting_resourcekey_by_score, axis=1,
                                                                    result_type='expand')
    df['rank'] = df.apply(get_rank, axis=1, result_type='expand')
    df = cal_recall(df, top_k_list)
    df = cal_mrr(df, top_k_list)
    df.to_csv('./metric.csv', index=False)
    for top_k in tqdm(top_k_list):
        metric[f'Recall@{top_k}'] = df[f'Recall@{top_k}'].sum() / df.shape[0]
        metric[f'MRR@{top_k}'] = df[f'MRR@{top_k}'].sum() / df.shape[0]
    return metric
```

### 2-2) DataLoader를 공유 하지 않는 경우
- `2-1` 부분에 추가적으로 5개의 함수를 Custom 및 추가해야함

|     Name     | Description                                                                                                      |
|:----------------:|:-----------------------------------------------------------------------------------------------------------------|
|       `maybe_log_save_evaluate`       | Train과 다른 형태의 Evaluation을 진행하기 위해 Custom Evaluation이 시작하는 곳                                                                                           |
|     `_rerank_evaluate`      | `rerank_evaluate` 함수를 호출                                                                          |
| `rerank_evaluate` | Custom Dataloader `get_rerank_eval_dataloader`함수를 통해 호출하고, `rerank_evaluation_loop` 결과값을 Wandb에 Logging                |
| `rerank_evaluation_loop` | Iteration 단위의 Model 예측을 하고, DDP로 예측된 값을 하나로 모으고 예측된 Logit을 compute_metrics에 명시된 평가지표에 따라 값을 산출 후 반환                |
| `get_rerank_eval_dataloader` | Custom Dataset을 Custom Dataloader로 변환하는 함수 |

- 호출 순서
&rarr; `_maybe_log_save_evaluate` &rarr; `_rerank_evaluate` &rarr; `rerank_evaluate` &rarr; `get_rerank_eval_dataloader` &rarr; `rerank_evaluation_loop` &rarr; `compute_metrics` &rarr; Complete
