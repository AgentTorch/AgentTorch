# Runs


## MiPro

Expt 2: Sees text descriptions

logs: ./expt2_models/logs/tensorboard/job_knowledge_exp_1756633744
Resolved slots: True
Data Seed: 42
Hold Out Test: 100

============================================================
INFO:__main__:MIPRO OPTIMIZATION COMPLETE
INFO:__main__:============================================================
INFO:__main__:Final Test Results (logged to MLflow):
INFO:__main__:  Test Samples: 100
INFO:__main__:  Average Optimized Score: 9458.737
INFO:__main__:  Average Baseline Score: 9413.838
MSE = 541.263
INFO:__main__:  Average Improvement: 44.898



Run 2: with 0 shot examples

============================================================
INFO:__main__:MIPRO OPTIMIZATION COMPLETE
INFO:__main__:============================================================
INFO:__main__:Final Test Results (logged to MLflow):
INFO:__main__:  Test Samples: 100
INFO:__main__:  Average Optimized Score: 9396.853
INFO:__main__:  Average Baseline Score: 9427.575
INFO:__main__:  Average Improvement: -30.721

```
"dspy_optimization": {
      "total_requests": 1,
      "total_tokens": 3303,
      "total_prompt_tokens": 1322,
      "total_completion_tokens": 1981,
      "total_cost_usd": 0.005349100000000001,
      "average_tokens_per_request": 3303.0,
      "average_cost_per_request": 0.005349100000000001,
      "models_used": {
        "gemini/gemini-2.5-flash": 1
      },
      "providers_used": {
        "google": 1
      },
      "first_request": "2025-09-02 23:14:57.036244",
      "last_request": "2025-09-02 23:14:57.036244",
      "prompt_tokens_mean": 1322.0,
      "prompt_tokens_std": 0.0,
      "prompt_tokens_min": 1322,
      "prompt_tokens_max": 1322,
      "prompt_tokens_p25": 1322.0,
      "prompt_tokens_p50": 1322.0,
      "prompt_tokens_p75": 1322.0,
      "prompt_tokens_p90": 1322.0,
      "prompt_tokens_p95": 1322.0,
      "prompt_tokens_p99": 1322.0,
      "completion_tokens_mean": 1981.0,
      "completion_tokens_std": 0.0,
      "completion_tokens_min": 1981,
      "completion_tokens_max": 1981,
      "completion_tokens_p25": 1981.0,
      "completion_tokens_p50": 1981.0,
      "completion_tokens_p75": 1981.0,
      "completion_tokens_p90": 1981.0,
      "completion_tokens_p95": 1981.0,
      "completion_tokens_p99": 1981.0
    },
    "eval": {
      "total_requests": 206,
      "total_tokens": 661369,
      "total_prompt_tokens": 262172,
      "total_completion_tokens": 399197,
      "total_cost_usd": 1.0766441000000009,
      "average_tokens_per_request": 3210.529126213592,
      "average_cost_per_request": 0.005226427669902917,
      "models_used": {
        "gemini/gemini-2.5-flash": 206
      },
      "providers_used": {
        "google": 206
      },
      "first_request": "2025-09-02 23:15:05.676608",
      "last_request": "2025-09-02 23:52:06.069790",
      "prompt_tokens_mean": 1272.6796116504854,
      "prompt_tokens_std": 68.98616909252735,
      "prompt_tokens_min": 1045,
      "prompt_tokens_max": 1373,
      "prompt_tokens_p25": 1275.0,
      "prompt_tokens_p50": 1288.0,
      "prompt_tokens_p75": 1305.0,
      "prompt_tokens_p90": 1320.0,
      "prompt_tokens_p95": 1332.0,
      "prompt_tokens_p99": 1340.0,
      "completion_tokens_mean": 1937.8495145631068,
      "completion_tokens_std": 395.3083614788639,
      "completion_tokens_min": 723,
      "completion_tokens_max": 3247,
      "completion_tokens_p25": 1662.25,
      "completion_tokens_p50": 1933.0,
      "completion_tokens_p75": 2166.5,
      "completion_tokens_p90": 2444.0,
      "completion_tokens_p95": 2625.0,
      "completion_tokens_p99": 2866.2
    },
```

Run 3: mipro2_light_100_b_1756942502

============================================================
INFO:__main__:MIPRO OPTIMIZATION COMPLETE
INFO:__main__:============================================================
INFO:__main__:Final Test Results (logged to MLflow):
INFO:__main__:  Test Samples: 100
INFO:__main__:  Average Optimized Score: 8752.082
INFO:__main__:  Average Baseline Score: 8757.450
INFO:__main__:  Average Improvement: -5.368
INFO:__main__:


```
"operation_stats": {
    "dspy_optimization": {
      "total_requests": 1,
      "total_tokens": 3303,
      "total_prompt_tokens": 1322,
      "total_completion_tokens": 1981,
      "total_cost_usd": 0.005349100000000001,
      "average_tokens_per_request": 3303.0,
      "average_cost_per_request": 0.005349100000000001,
      "models_used": {
        "gemini/gemini-2.5-flash": 1
      },
      "providers_used": {
        "google": 1
      },
      "first_request": "2025-09-03 16:35:17.044708",
      "last_request": "2025-09-03 16:35:17.044708",
      "prompt_tokens_mean": 1322.0,
      "prompt_tokens_std": 0.0,
      "prompt_tokens_min": 1322,
      "prompt_tokens_max": 1322,
      "prompt_tokens_p25": 1322.0,
      "prompt_tokens_p50": 1322.0,
      "prompt_tokens_p75": 1322.0,
      "prompt_tokens_p90": 1322.0,
      "prompt_tokens_p95": 1322.0,
      "prompt_tokens_p99": 1322.0,
      "completion_tokens_mean": 1981.0,
      "completion_tokens_std": 0.0,
      "completion_tokens_min": 1981,
      "completion_tokens_max": 1981,
      "completion_tokens_p25": 1981.0,
      "completion_tokens_p50": 1981.0,
      "completion_tokens_p75": 1981.0,
      "completion_tokens_p90": 1981.0,
      "completion_tokens_p95": 1981.0,
      "completion_tokens_p99": 1981.0
    },
    "train": {
      "total_requests": 49,
      "total_tokens": 298443,
      "total_prompt_tokens": 201725,
      "total_completion_tokens": 96718,
      "total_cost_usd": 0.30231250000000004,
      "average_tokens_per_request": 6090.673469387755,
      "average_cost_per_request": 0.006169642857142858,
      "models_used": {
        "gemini/gemini-2.5-flash": 49
      },
      "providers_used": {
        "google": 49
      },
      "first_request": "2025-09-03 16:35:50.468422",
      "last_request": "2025-09-03 16:43:47.860598",
      "prompt_tokens_mean": 4116.836734693878,
      "prompt_tokens_std": 1860.2843248446748,
      "prompt_tokens_min": 467,
      "prompt_tokens_max": 8857,
      "prompt_tokens_p25": 3655.0,
      "prompt_tokens_p50": 3941.0,
      "prompt_tokens_p75": 4377.0,
      "prompt_tokens_p90": 6767.000000000001,
      "prompt_tokens_p95": 8052.999999999996,
      "prompt_tokens_p99": 8738.919999999998,
      "completion_tokens_mean": 1973.8367346938776,
      "completion_tokens_std": 826.3802639774128,
      "completion_tokens_min": 501,
      "completion_tokens_max": 3998,
      "completion_tokens_p25": 1350.0,
      "completion_tokens_p50": 1880.0,
      "completion_tokens_p75": 2469.0,
      "completion_tokens_p90": 3342.0,
      "completion_tokens_p95": 3526.3999999999996,
      "completion_tokens_p99": 3860.239999999999
    },
    "eval": {
      "total_requests": 192,
      "total_tokens": 918710,
      "total_prompt_tokens": 533851,
      "total_completion_tokens": 384859,
      "total_cost_usd": 1.1223027999999995,
      "average_tokens_per_request": 4784.947916666667,
      "average_cost_per_request": 0.005845327083333331,
      "models_used": {
        "gemini/gemini-2.5-flash": 192
      },
      "providers_used": {
        "google": 192
      },
      "first_request": "2025-09-03 16:44:01.841951",
      "last_request": "2025-09-03 17:19:51.341217",
      "prompt_tokens_mean": 2780.4739583333335,
      "prompt_tokens_std": 1510.455949839374,
      "prompt_tokens_min": 1045,
      "prompt_tokens_max": 4587,
      "prompt_tokens_p25": 1288.0,
      "prompt_tokens_p50": 2705.5,
      "prompt_tokens_p75": 4303.0,
      "prompt_tokens_p90": 4323.8,
      "prompt_tokens_p95": 4336.35,
      "prompt_tokens_p99": 4350.45,
      "completion_tokens_mean": 2004.4739583333333,
      "completion_tokens_std": 386.49219665063305,
      "completion_tokens_min": 723,
      "completion_tokens_max": 3247,
      "completion_tokens_p25": 1759.25,
      "completion_tokens_p50": 1995.0,
      "completion_tokens_p75": 2256.75,
      "completion_tokens_p90": 2461.8,
      "completion_tokens_p95": 2667.8999999999996,
      "completion_tokens_p99": 2868.0000000000014
    }
```


## A3O

Expt 1: p3o_b20_e2_1756698775
logs: ./expt2_models/logs/tensorboard/job_knowledge_exp_1756633744
Resolved slots: True
Data Seed: 42
Batch Size: 20
Hold Out Test: 100


2025-09-01 02:58:44,024 - INFO - Final Test Results:
2025-09-01 02:58:44,024 - INFO -   Test Samples: 100
2025-09-01 02:58:44,024 - INFO -   Average Test Reward: 9347.429
2025-09-01 02:58:44,024 - INFO -   Average Test MSE: 652.571

============================================================
TOKEN USAGE SUMMARY - Session: p3o_b20_e2_1756698775 : TRAIN ONLY
============================================================
Duration: 5:48:28.592776
Total Requests: 3,997
Total Tokens: 5,869,156
  - Prompt Tokens: 4,905,374
  - Completion Tokens: 963,782

============================================================
TOKEN USAGE SUMMARY - Session: p3o_b20_e2_1756698775 : TRAIN + TEST
============================================================
Duration: 6:05:22.461146
Total Requests: 4,097
Total Tokens: 6,015,942
  - Prompt Tokens: 5,028,029
  - Completion Tokens: 987,913
============================================================