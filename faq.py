# faq.py  
'''
FAQ库查询和相似度计算
'''
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
import json

def compute_similarity(intent_vector, faq_vectors):  
    # 计算意图与FAQ库中问题的相似度  
    similarities = cosine_similarity(intent_vector, faq_vectors)  
    return similarities  

def get_top_answers(similarities, faq_data, top_k=3):  
    # 获取相似度最高的K个答案  
    top_indices = similarities.argsort()[0][-top_k:][::-1]  
    answers = [faq_data[i]['answer'] for i in top_indices]  
    return answers


def load_faq():
    '''
     将 faq.json中的所有json行抽取成一个List[str]
     
    '''
    faq_data = []
    with open('faq.json', 'r', encoding='utf8') as f:  
        for line in f:  
            faq_data.append(json.loads(line))
            
    return faq_data





if __name__ == '__main__':  
    # 示例数据  
    faq_data = load_faq()
    # 构建TF-IDF向量  
    tfidf_vectorizer = TfidfVectorizer()
    faq_vectors = tfidf_vectorizer.fit_transform(faq_data)
    print(faq_vectors)
    print(faq_vectors.shape)
    
    user_input = "用户输入"
    user_vector = tfidf_vectorizer.transform([user_input])
    
    similarities = compute_similarity(user_vector, faq_vectors)
    print(similarities)
    