# Machine Learning industry_level Case

## FDDC Challenge Introduction
This is an industry algorithm competition with rich bonus, which is designed for Chinese engineers and professional quantitative talents. 

For those speicalists, it still costs them two to three months to get the well-performed and stable model. 
So you don't need to worry if you fell totally in loss when you see these large amount of data.

You can use the commonly used machine learning methods for predictive analysis, select a part of the data you want to
use for some analysis and processing. 

Due to the limited level, our RA team will not provide the jupyter literature for reference. 

But for students who want to work in this area, this is very helpful in your future career.

## Data Description in Chinese
(You can also read the English version in 0-read me-EN.pdf)

FDDC2018金融算法挑战赛
奖金：￥840000   参赛队伍：2724	2 724
	已结束		2018-08-24	￥840000	2724
 

赛题描述
在股票市场大力提倡价值投资的背景下，准确预测公司未来营收，是理性投资者最重要的事情。买入盈利超预期的公司，避开盈利能力差的公司，才是投资的长久之道。按照定义，营业收入是企业在生产经营活动中，因销售产品或提供劳务而取得的各项收入，它关系到企业的生存和发展，对企业有重要的意义。
数据描述
本赛题用到的数据包括历史财务数据、宏观数据、行情数据、行业数据。各数据包含的主要字段的名词解析以及财务数据的中英文对照。
财务数据包括三张表，分别为资产负债表 Balance Sheet、利润表 Income Statement、现金流量表 Cash Flow Statement。其中，由于非金融上市公司、证券、银行、保险四大行业的财务报表在结构上存在差异，所以每个类别又分为4个相对应的文档（csv格式）。这三张表代表了一个公司全部的财务信息，三大财务报表分析是投资的基础。
资产负债表：代表一个公司的资产与负债及股东权益，资产负债表是所有表格的基础。
利润表：代表一个公司的利润来源，而净利润则直接影响资产负债表中股东权益的变化。
现金流量表：代表一个公司的现金流量，更代表资产负债表的变化。现金流量表是对资产负债表变化的解释。现金的变化最终反映到资产负债表的现金及等价物一项。而现金的变化源泉则是净利润。净利润经过“经营”、“投资”、“筹资”三项重要的现金变动转变为最终的现金变化。
![image](https://github.com/ButBueatiful/dotvim/raw/master/screenshots/Financial Statement.jpg)
 

宏观数据 Macro Industry 是指一系列宏观经济学的统计指标， 包括生产总值(GDP)、国民总收入（GNI）、劳动者报酬、消费水平等。宏观经济周期是影响周期性行业的关键因素之一，对上市公司的经营情况也有直接的影响。
行业数据 Industry Data 行业数据可以指示某个行业的发展态势，上市公司都会有自己所在的行业，分析行业的发展趋势、所处阶段等可对上市公司经营情况做出大体的判断（如从汽车行业每月的销量数据中，可以看到行业的景气程度）。
公司经营数据 Company Operation Data 一般为月度数据，代表特定公司主营业务月度的统计值，与公司营收密切相关，每个公司指标不一样。
行情数据 Market Data 行情数据代表上市公司股票月度交易行情，主要包括价格、成交量、成交额、换手率等。
提交说明
选手提交格式参考示例文件 FDDC_financial_submit.csv，包含两列数据，分别是公司代码和二季度预测营收；预测值以百万为单位，保留两位小数。

评估指标
参赛队伍需要提交指定公司的二季度营收数据， 以百万为单位，保留两位小数 。该结果将与真实财报发布的数值进行对比。计算各个公司的相对预测误差，并进行对数市值加权，计算公式如下：
![image](https://github.com/ButBueatiful/dotvim/raw/master/screenshots/Model Evaluation.jpg)
 
注意事项
1、本次比赛若使用外部数据，必须向其他参赛队公开。
2、如果抽查发现参赛队伍有造假和作弊行为，将取消该队伍的参赛资格。

