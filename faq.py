# faq.py  
'''
FAQ库查询和相似度计算
'''
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

def compute_similarity(intent_vector, faq_vectors):  
    # 计算意图与FAQ库中问题的相似度  
    similarities = cosine_similarity(intent_vector, faq_vectors)  
    return similarities  

def get_top_answers(similarities, faq_data, top_k=3):  
    # 获取相似度最高的K个答案  
    top_indices = similarities.argsort()[0][-top_k:][::-1]  
    answers = [faq_data[i]['answer'] for i in top_indices]  
    return answers