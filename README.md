# An-Academic-Paper-Chatbot-based-on-LLama3.1-and-Knowledge-Graph
基于知识图谱和大模型的对话系统



## 项目要求
_______
1. 所有模型配置，模型路径，超参数相关信息请放到一个单独的config.py文件里。
2. 所有数据集的加载和预处理工作请放到load.py文件里。
3. 模型的完整训练流程请放到main.py文件里。
4. 模型性能评估相关代码请放到evaluate.py文件里，包含一个Evaluator类。Evaluator类包含 eval, write_stats, show_stats, decode, write_stats_to_csv等成员函数。
5. 所有模型结构的设计，定义，请放到model.py文件中。
6. 所有数据集文件存放在项目目录的data目录下。
7. 所有模型权重存放在项目目录的model目录下。



## 项目内容如下：

项目环境：pytorch+Django+Docker
1. 项目描述: 整合数十篇AI论文的知识点来搭建作为一个FAQ知识库,为研究生和教授们提供一个论文问答服务。
2. 项目主要构成: 对话系统、论文上传, pdf解析+读取，论文转换为论文框架（GPT-4），论文框架分类和论文中的实体抽取, 以及问答服务实现。
3. 系统输入：用户上传的论文.pdf+问题
4. FAQ库的单条数据格式{"stand_query":"xxxx", "similar_query":set('xxx','xxx',...), "answer": xxxx}
5. 论文框架结构：{"Abstract":'xxxxxx', "Introduction":"xxxxxxx", "Methodology":"xxxxxx", "Experiment":'xxxxxx', "Results": 'xxxxxxx'}
6. 项目任务:
- 使用llama3-8B模型来针对论文框架集合进行微调,实现论文自动分类，分到12个类别: ["Attention & Model Architecture", "Benchmarks", "BERT", "Chain-of-Thought", "Fine-Tuning", "Long-Context", "LoRA", "Instruction&Prompt-Tuning", "RAG", "RL", "RLHF", "Reasoning"]。
- 使用Bert+BiLSTM+CRF抽取论文中实体和关系,构建论文框架知识图谱
- 识别老师和学生提交的问题的意图【在对话系统部署前，使用该系统的历史问答数据【问题-意图】对llama3进行微调，生成意图分类模型】，将这个意图文本转为向量，和FAQ库中的问题来计算相似度。计算方法为：意图与standard_query计算相似度（score1），再与similar_query问题集中的所有问题计算相似度再取平均值(score2), 再计算当前论文的预测类别与FAQ中的样本的类别的相似度（score3）。最后将这3个分数以 5:5:10 的比例做加权和, 得到ranking_score, 计算当前意图与FAQ库中的每个样本的ranking_Score, 取分数最高的3条，将其中的Answer提取出来组合到一起：“Answer1: xxx. Answer2: xxxx. Answer3: xxxx”。
- 根据用意图查询FAQ库得到的回答，再用意图去查询知识图谱并返回比较匹配的{实体、关系、属性、类型、实例、量化信息、描述}（以字典形式返回）。再把FAQ的回答和知识图谱的返回信息组合成一个嵌套字典：meta_dict = {意图：{Answer1: xxx, Answer2:xxx, Answer3:xxx},  实体：xxx, 关系：xxx ...}。再把meta_dict作为输入给到GPT4o, 最后获取响应给到用户，同时将当前的响应，query内容、论文类别，加入到FAQ库中。
使用Docker封装查询结果并提供Rest API接口对外提供服务






## 项目流程总结
用户通过前端界面上传论文PDF和提问的问题。
服务器端解析PDF，提取文本内容。
调用GPT-4，将论文文本转换为论文框架结构。
使用LLama3-8B模型对论文框架进行分类，预测论文类别。
使用Bert+BiLSTM+CRF模型从论文中抽取实体和关系，构建知识图谱。
识别用户问题的意图，将其转换为向量。
计算意图与FAQ库和论文类别的相似度，获取排名前3的答案。
查询知识图谱，获取相关的实体、关系等信息。
将FAQ答案和知识图谱信息组合成meta_dict，并作为输入提供给GPT-4，生成最终的回答。
将回答、用户问题和论文类别添加到FAQ库中。
使用Docker封装整个服务，提供REST API接口供外部调用。