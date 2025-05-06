       
这个API对象是一个封装了与后端服务交互的JavaScript模块，主要功能如下：

1. **接收内容**：
- 接收各种API请求参数，包括：
  - 问题文本（question）
  - 论文ID（paperId）
  - 文件对象（file）
  - 搜索关键词（keyword）
  - 实体类型（entityType）等
- 对于文件上传还接收进度回调函数（progressCallback）

1. **返回内容**：
- 所有方法都返回Promise对象
- 解析后的JSON数据（通过response.json()）
- 错误时会抛出Error对象

1. **外部使用方式**：
```javascript
// 1. 初始化后直接使用API对象
API.askQuestion("什么是注意力机制？")
  .then(response => console.log(response))
  .catch(error => console.error(error));

// 2. 文件上传带进度回调
API.uploadFile(fileObject, (progress) => {
  console.log(`上传进度: ${progress}%`);
});

// 3. 获取知识图谱数据
API.getKnowledgeGraphData("author")
  .then(data => visualizeGraph(data));

// 4. 其他API调用方式类似
```

主要特点：
1. 自执行函数(IIFE)封装，立即创建单例对象
2. 统一错误处理机制
3. 支持GET/POST/DELETE等多种HTTP方法
4. 文件上传使用XMLHttpRequest实现进度监控
5. 所有方法都是异步的，返回Promise

使用时需要注意：
1. 需要处理Promise的resolve和reject
2. 文件上传必须提供有效的File对象
3. BASE_URL可能需要根据部署环境修改
4. 错误处理建议使用catch或try/catch语法

        