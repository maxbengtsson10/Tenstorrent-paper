# All scores are computed using a fixed window size of 512 and 5 runs

| Compute Type | Compute Name                       | Tokens/Sec | Per Run Time | Total Time |
| :----------- | :--------------------------------- | :--------: | :----------: | :--------: |
| CPU          | Apple M3 Max CPU (14 core)         | 8.00       | 63.89        | 319.46     |
| CPU          | Intel                              |            |              |            |
| GPU          | Apple M3 Max GPU (30 core)         | 41.61      | 12.28        | 61.40      |
| GPU          | Nvidia T4                          | 25.86      | 19.76        | 98.79      |
| GPU          | Nvidia L4                          | 52.82      | 9.68         | 48.38      |
| GPU          | Nvidia A100                        | 78.30      | 6.53         | 32.63      |
| GPU          | Nvidia A6000                       | 84.94      | 6.02         | 30.76      |
| TPU          | Google TPU v3-8 (Kaggle) (Dynamic) | 3.49       | 146.69       | -          |
| TPU          | Google TPU v6e-1 (Colab) (Dynamic) | 1.81       | 282.19       | -          |
| TPU          | Google TPU v6e-1 (Colab) (Fixed)   | 11.77      | 43.49        | -          |