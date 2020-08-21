
## Aliyun project:

### Introduction
This dataset contains two commercial targeting campaign logs in Alipay. Due to privacy issue, data is sampled and desensitized. Although the statistical results on this data set deviate from the actual scale of Alipay.com, it will not affect the applicability of the solution.


### Description
emb_tb_2.csv: User feature dataset.
effect_tb.csv: Click/Non-click dataset.
seed_cand_tb.csv: Seed users and candidate users dataset.
Fields	Descriptions
dmp_id	The unique ID of a targeting campaign.
user_id	The unique ID of an Alipay user.
role	Value from {'seed', 'cand'}. 'seed' indicates this user_id is chosen as a seed for expansion. 'cand' indicates this user_id is a candidate user for this campaign, and could be chosen.
label	Denotes whether a user clicked the campaign ads in that day dt.
dt	Values from {1,2}. Indicates whether it's a first day log (“1”) or a second day log (“2”) for the target campaign.
emb	A 16-dimensional embedding generated from the user's raw profile features and activity features using graph embedding techniques.

## Reference
https://tianchi.aliyun.com/dataset/dataDetail?dataId=50893

