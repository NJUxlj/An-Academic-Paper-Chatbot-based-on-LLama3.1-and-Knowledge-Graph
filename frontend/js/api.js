/**
 * API模块，负责与后端服务交互
 */
const API = (() => {
    const BASE_URL = 'http://localhost:8000';  // 根据实际部署情况修改
    
    /**
     * 发送HTTP请求
     * @param {string} endpoint - API端点【API路径】
     * @param {string} method - HTTP方法【默认为GET】
     * @param {Object} data - 请求数据【默认为null】
     * @returns {Promise} - 响应Promise
     */
    const request = async (endpoint, method = 'GET', data = null) => {  // // 定义异步箭头函数,  使用 async 关键字表示返回Promise
        const url = `${BASE_URL}${endpoint}`;
        
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data) {
            if (method === 'GET') {
                const params = new URLSearchParams(data); // 通过URLSearchParams转换为查询字符串
                url = `${url}?${params}`;
            } else {
                options.body = JSON.stringify(data);
            }
        }
        
        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {  // 检查响应状态码(!response.ok)
                const errorData = await response.json();   // 返回的是一个 Promise 解析后的 JavaScript 对象，包含从 JSON 反序列化得到的数据
                throw new Error(errorData.detail || '请求失败');
            }
            
            return await response.json();
        } catch (error) {
            console.error('API请求错误:', error);
            throw error;
        }
    };
    
    /**
     * 上传文件
     * @param {File} file - 要上传的文件
     * @param {Function} progressCallback - 进度回调函数
     * @returns {Promise} - 响应Promise
     */
    const uploadFile = async (file, progressCallback) => {   // 定义异步箭头函数,  使用 async 关键字表示返回Promise
        const url = `${BASE_URL}/upload_paper`;
        
        const formData = new FormData();   // 创建FormData对象用于文件上传
        formData.append('file', file);   // 添加文件到表单数据
        
        // 创建一个Promise来处理文件上传
        return new Promise((resolve, reject) => {  // 返回Promise对象实现异步控制流
            // 使用XMLHttpRequest对象实现文件上传
            const xhr = new XMLHttpRequest();
            // 初始化POST请求（异步）
            xhr.open('POST', url, true);


            // - xhr.upload.onprogress ：XMLHttpRequest的上传进度事件监听器
            // - event 参数包含上传进度信息：
            // - lengthComputable ：布尔值，表示是否可计算进度
            // - loaded ：已上传字节数
            // - total ：总字节数
            
            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable && progressCallback) {
                    const progress = (event.loaded / event.total) * 100;
                    progressCallback(progress); // 调用进度回调函数
                }
            };

            // - xhr.onload ：XMLHttpRequest请求完成时触发的事件
            // - 箭头函数语法： () => {...}， 相当于python里的lambda函数
            
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {   // 检查HTTP状态码是否在成功范围(200-299)
                    // - 解析响应文本为JSON
                    // - 通过Promise的resolve返回数据
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    try {
                        const errorData = JSON.parse(xhr.responseText);
                        reject(new Error(errorData.detail || '上传失败'));  // 通过Promise的reject返回错误信息
                    } catch (e) {
                        reject(new Error('上传失败'));
                    }
                }
            };
            
            xhr.onerror = () => {
                reject(new Error('网络错误'));
            };
            
            xhr.send(formData);
        });
    };
    
    // 问答API
    const askQuestion = (question, paperId = null) => {
        return request('/ask', 'POST', { question, paper_id: paperId });
    };
    
    // 获取论文列表
    const getPapers = (category = null, limit = 10) => {
        const params = { limit };
        if (category) params.category = category;
        
        return request('/papers', 'GET', params);
    };
    
    // 获取论文分类列表
    const getCategories = () => {
        return request('/categories', 'GET');
    };
    
    // 获取论文推荐
    const getRecommendations = (paperId = null, question = null) => {
        const data = { max_recommendations: 5 };
        
        if (paperId) data.paper_id = paperId;
        if (question) data.question = question;
        
        return request('/recommend', 'POST', data);
    };
    
    // 删除论文
    const deletePaper = (paperId) => {
        return request(`/paper/${paperId}`, 'DELETE');
    };
    
    // 获取知识图谱统计信息
    const getKnowledgeGraphStats = () => {
        return request('/kg/stats', 'GET');
    };
    
    // 搜索实体
    const searchEntities = (keyword, entityType = null) => {
        const params = { keyword };
        if (entityType) params.entity_type = entityType;
        
        return request('/kg/entities/search', 'GET', params);
    };
    
    // 获取实体详情
    const getEntityDetails = (entityName, entityType = null) => {
        const params = { name: entityName };
        if (entityType) params.type = entityType;
        
        return request('/kg/entity', 'GET', params);
    };
    
    // 获取知识图谱数据
    const getKnowledgeGraphData = (entityType = null) => {
        const params = {};
        if (entityType) params.entity_type = entityType;
        
        return request('/kg/data', 'GET', params);
    };
    
    // 系统健康检查
    const healthCheck = () => {
        return request('/health', 'GET');
    };
    
    return {
        askQuestion,
        getPapers,
        getCategories,
        getRecommendations,
        deletePaper,
        uploadFile,
        getKnowledgeGraphStats,
        searchEntities,
        getEntityDetails,
        getKnowledgeGraphData,
        healthCheck
    };
})();
