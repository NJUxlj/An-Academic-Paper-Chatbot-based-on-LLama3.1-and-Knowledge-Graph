/**
 * 聊天模块，负责处理聊天界面和逻辑
 */
const ChatModule = (() => {
    // 元素引用
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const clearChatButton = document.getElementById('clear-chat');
    const exportChatButton = document.getElementById('export-chat');
    const recommendationsContainer = document.getElementById('recommendations');
    
    // 当前对话状态
    let currentPaperId = null;
    let currentPaperTitle = '';
    let chatHistory = [];
    let isWaitingForResponse = false;
    
    // 初始化
    const init = () => {
        // 添加欢迎消息
        addAssistantMessage('您好，我是学术论文助手，可以帮您解答关于AI论文的问题。您可以询问特定论文内容、研究方法、模型架构等问题。请问有什么可以帮您的？');
        
        // 绑定事件处理器
        bindEventHandlers();
    };
    
    // 绑定事件处理器
    const bindEventHandlers = () => {
        // 发送按钮点击
        sendButton.addEventListener('click', handleSendMessage);
        
        // 输入框回车键
        chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSendMessage();
            }
        });
        
        // 清空聊天按钮
        clearChatButton.addEventListener('click', clearChat);
        
        // 导出聊天按钮
        exportChatButton.addEventListener('click', exportChat);
    };
    
    // 处理发送消息
    const handleSendMessage = () => {
        const message = chatInput.value.trim();
        
        if (!message || isWaitingForResponse) {
            return;
        }
        
        // 添加用户消息到界面
        addUserMessage(message);
        
        // 清空输入框
        chatInput.value = '';
        
        // 显示加载指示器
        showTypingIndicator();
        
        // 设置等待状态
        isWaitingForResponse = true;
        
        // 发送请求到API
        API.askQuestion(message, currentPaperId)
            .then(response => {
                // 移除加载指示器
                hideTypingIndicator();
                
                // 添加回答
                addAssistantMessage(response.answer);
                
                // 更新推荐
                if (response.recommendations && response.recommendations.length > 0) {
                    updateRecommendations(response.recommendations);
                }
                
                // 恢复状态
                isWaitingForResponse = false;
            })
            .catch(error => {
                // 移除加载指示器
                hideTypingIndicator();
                
                // 显示错误消息
                addAssistantMessage(`抱歉，发生了错误：${error.message || '未知错误'}`);
                
                // 恢复状态
                isWaitingForResponse = false;
            });
    };
    
    // 添加用户消息
    const addUserMessage = (text) => {
        const messageElement = createMessageElement(text, 'user-message');
        chatMessages.appendChild(messageElement);
        
        // 滚动到底部
        scrollToBottom();
        
        // 添加到历史记录
        chatHistory.push({
            role: 'user',
            content: text,
            timestamp: new Date().toISOString()
        });
    };
    
    // 添加助手消息
    const addAssistantMessage = (text) => {
        const messageElement = createMessageElement(text, 'assistant-message');
        chatMessages.appendChild(messageElement);
        
        // 滚动到底部
        scrollToBottom();
        
        // 添加到历史记录
        chatHistory.push({
            role: 'assistant',
            content: text,
            timestamp: new Date().toISOString()
        });
    };
    
    // 创建消息元素
    const createMessageElement = (text, className) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${className}`;
        
        // 处理文本中的Markdown（简单实现）
        const formattedText = formatMarkdown(text);
        messageDiv.innerHTML = formattedText;
        
        // 添加时间戳
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = getFormattedTime();
        messageDiv.appendChild(timeDiv);
        
        return messageDiv;
    };
    
    // 格式化Markdown文本（简单实现）
    const formatMarkdown = (text) => {
        // 粗体
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // 斜体
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // 代码块
        text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // 行内代码
        text = text.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // 换行
        text = text.replace(/\n/g, '<br>');
        
        return text;
    };
    
    // 显示打字指示器
    const showTypingIndicator = () => {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'chat-message assistant-message typing-indicator-container';
        indicatorDiv.id = 'typing-indicator';
        
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        
        // 添加三个点
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingIndicator.appendChild(dot);
        }
        
        indicatorDiv.appendChild(typingIndicator);
        chatMessages.appendChild(indicatorDiv);
        
        // 滚动到底部
        scrollToBottom();
    };
    
    // 隐藏打字指示器
    const hideTypingIndicator = () => {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    };
    
    // 滚动到底部
    const scrollToBottom = () => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };
    
    // 获取格式化时间
    const getFormattedTime = () => {
        return moment().format('HH:mm');
    };
    
    // 清空聊天
    const clearChat = () => {
        // 确认是否清空
        if (confirm('确定要清空聊天记录吗？')) {
            // 保留最初的欢迎消息
            const welcomeMessage = chatHistory[0];
            
            // 清空聊天记录
            chatMessages.innerHTML = '';
            chatHistory = [];
            
            // 重新添加欢迎消息
            if (welcomeMessage && welcomeMessage.role === 'assistant') {
                addAssistantMessage(welcomeMessage.content);
            } else {
                addAssistantMessage('您好，我是学术论文助手，可以帮您解答关于AI论文的问题。您可以询问特定论文内容、研究方法、模型架构等问题。请问有什么可以帮您的？');
            }
            
            // 清空推荐
            updateRecommendations([]);
        }
    };
    
    // 导出聊天
    const exportChat = () => {
        // 准备导出数据
        const exportData = {
            timestamp: new Date().toISOString(),
            paperInfo: currentPaperId ? {
                id: currentPaperId,
                title: currentPaperTitle
            } : null,
            history: chatHistory
        };
        
        // 转换为JSON字符串
        const jsonString = JSON.stringify(exportData, null, 2);
        
        // 创建Blob
        const blob = new Blob([jsonString], { type: 'application/json' });
        
        // 创建下载链接
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_export_${new Date().getTime()}.json`;
        
        // 触发下载
        document.body.appendChild(a);
        a.click();
        
        // 清理
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
    
    // 更新推荐列表
    const updateRecommendations = (recommendations) => {
        // 清空现有推荐
        recommendationsContainer.innerHTML = '';
        
        if (!recommendations || recommendations.length === 0) {
            recommendationsContainer.innerHTML = '<p class="text-muted">基于您的问题，我们会在这里显示相关论文推荐。</p>';
            return;
        }
        
        // 创建推荐列表
        const listElement = document.createElement('ul');
        listElement.className = 'list-group';
        
        recommendations.forEach(paper => {
            const item = document.createElement('li');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            const titleDiv = document.createElement('div');
            titleDiv.innerHTML = `<strong>${paper.title}</strong><br><small>${paper.author || '未知作者'}</small>`;
            
            const badgeDiv = document.createElement('div');
            if (paper.category) {
                const badge = document.createElement('span');
                badge.className = 'badge bg-primary rounded-pill';
                badge.textContent = paper.category;
                badgeDiv.appendChild(badge);
            }
            
            item.appendChild(titleDiv);
            item.appendChild(badgeDiv);
            
            // 添加点击事件
            item.addEventListener('click', () => {
                setCurrentPaper(paper.paper_id, paper.title);
                addAssistantMessage(`已切换到论文《${paper.title}》，您可以开始询问该论文的相关问题。`);
            });
            
            listElement.appendChild(item);
        });
        
        recommendationsContainer.appendChild(listElement);
    };
    
    // 设置当前论文
    const setCurrentPaper = (paperId, paperTitle) => {
        currentPaperId = paperId;
        currentPaperTitle = paperTitle;
        
        // 更新当前论文显示
        updateCurrentPaperDisplay();
    };
    
    // 更新当前论文显示
    const updateCurrentPaperDisplay = () => {
        const currentPaperContainer = document.getElementById('current-paper');
        
        if (currentPaperId) {
            currentPaperContainer.innerHTML = `
                <div class="card-body">
                    <h6 class="card-title">当前聊天的论文</h6>
                    <p class="paper-title">${currentPaperTitle}</p>
                    <button class="btn btn-sm btn-outline-secondary" id="clear-paper-btn">
                        <i class="fas fa-times me-1"></i>清除
                    </button>
                </div>
            `;
            
            // 添加清除按钮事件
            document.getElementById('clear-paper-btn').addEventListener('click', clearCurrentPaper);
        } else {
            currentPaperContainer.innerHTML = `
                <p class="text-muted">选择一篇论文以关注特定内容。</p>
                <button class="btn btn-sm btn-outline-primary" id="select-paper-btn">
                    <i class="fas fa-book me-1"></i>选择论文
                </button>
            `;
            
            // 添加选择论文按钮事件
            document.getElementById('select-paper-btn').addEventListener('click', () => {
                // 显示选择论文模态框
                const selectPaperModal = new bootstrap.Modal(document.getElementById('select-paper-modal'));
                selectPaperModal.show();
            });
        }
    };
    
    // 清除当前论文
    const clearCurrentPaper = () => {
        currentPaperId = null;
        currentPaperTitle = '';
        
        // 更新显示
        updateCurrentPaperDisplay();
        
        // 通知用户
        addAssistantMessage('已清除当前论文，您可以询问一般性问题或选择新的论文。');
    };
    
    // 公开方法
    return {
        init,
        addUserMessage,
        addAssistantMessage,
        setCurrentPaper,
        clearChat
    };
})();

// 在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    ChatModule.init();
});
