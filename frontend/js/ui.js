/**
 * UI模块，负责处理界面切换和全局UI操作
 */
const UIModule = (() => {
    // 元素引用
    const chatTab = document.getElementById('chat-tab');
    const papersTab = document.getElementById('papers-tab');
    const uploadTab = document.getElementById('upload-tab');
    const kgTab = document.getElementById('kg-tab');
    
    const chatView = document.getElementById('chat-view');
    const papersView = document.getElementById('papers-view');
    const uploadView = document.getElementById('upload-view');
    const kgView = document.getElementById('kg-view');
    
    const selectPaperModal = document.getElementById('select-paper-modal');
    
    // 当前激活的标签
    let activeTab = 'chat';
    
    // 初始化
    const init = () => {
        // 绑定事件处理器
        bindEventHandlers();
        
        // 检查系统状态
        checkSystemStatus();
    };
    
    // 绑定事件处理器
    const bindEventHandlers = () => {
        // 标签点击事件
        chatTab.addEventListener('click', (event) => {
            event.preventDefault();
            switchToTab('chat');
        });
        
        papersTab.addEventListener('click', (event) => {
            event.preventDefault();
            switchToTab('papers');
        });
        
        uploadTab.addEventListener('click', (event) => {
            event.preventDefault();
            switchToTab('upload');
        });
        
        kgTab.addEventListener('click', (event) => {
            event.preventDefault();
            switchToTab('kg');
        });
        
        // 选择论文模态框显示事件
        selectPaperModal.addEventListener('show.bs.modal', () => {
            // 加载论文选择表格
            PapersModule.loadSelectPaperTable();
        });
    };
    
    // 切换到指定标签
    const switchToTab = (tabName) => {
        // 更新标签状态
        updateTabStatus(tabName);
        
        // 保存当前标签
        activeTab = tabName;
        
        // 调整特定视图
        adjustSpecificView(tabName);
    };
    
    // 更新标签状态
    const updateTabStatus = (tabName) => {
        // 清除所有标签激活状态
        [chatTab, papersTab, uploadTab, kgTab].forEach(tab => {
            tab.classList.remove('active');
        });
        
        // 隐藏所有视图
        [chatView, papersView, uploadView, kgView].forEach(view => {
            view.classList.remove('active');
        });
        
        // 激活选中的标签和视图
        switch (tabName) {
            case 'chat':
                chatTab.classList.add('active');
                chatView.classList.add('active');
                break;
                
            case 'papers':
                papersTab.classList.add('active');
                papersView.classList.add('active');
                break;
                
            case 'upload':
                uploadTab.classList.add('active');
                uploadView.classList.add('active');
                break;
                
            case 'kg':
                kgTab.classList.add('active');
                kgView.classList.add('active');
                break;
        }
    };
    
    // 调整特定视图
    const adjustSpecificView = (tabName) => {
        // 知识图谱视图需要特殊处理，确保图正确渲染
        if (tabName === 'kg') {
            // 窗口大小可能影响图的布局，需要刷新
            setTimeout(() => {
                // 如果KnowledgeGraphModule已初始化
                if (KnowledgeGraphModule && typeof KnowledgeGraphModule.loadKnowledgeGraph === 'function') {
                    KnowledgeGraphModule.loadKnowledgeGraph();
                }
            }, 100);
        }
    };
    
    // 显示通知
    const showNotification = (message, type = 'info') => {
        // 创建通知元素
        const notificationDiv = document.createElement('div');
        notificationDiv.className = `toast align-items-center text-white bg-${type}`;
        notificationDiv.setAttribute('role', 'alert');
        notificationDiv.setAttribute('aria-live', 'assertive');
        notificationDiv.setAttribute('aria-atomic', 'true');
        
        notificationDiv.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        // 添加到通知容器
        // 如果页面中没有通知容器，则创建一个
        let toastContainer = document.querySelector('.toast-container');
        
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(notificationDiv);
        
        // 初始化Bootstrap toast
        const toast = new bootstrap.Toast(notificationDiv, {
            delay: 5000
        });
        
        // 显示通知
        toast.show();
        
        // 通知关闭后移除元素
        notificationDiv.addEventListener('hidden.bs.toast', () => {
            notificationDiv.remove();
            
            // 如果容器为空，也移除容器
            if (toastContainer.children.length === 0) {
                toastContainer.remove();
            }
        });
    };
    
    // 显示确认对话框
    const showConfirmDialog = (message, callback) => {
        if (confirm(message)) {
            callback(true);
        } else {
            callback(false);
        }
    };
    
    // 检查系统状态
    const checkSystemStatus = () => {
        // 发送健康检查请求
        API.healthCheck()
            .then(response => {
                if (response.status === 'ok') {
                    showNotification('系统正常运行中', 'success');
                } else {
                    showNotification('系统状态异常', 'warning');
                }
            })
            .catch(error => {
                console.error('系统状态检查失败:', error);
                showNotification('无法连接到服务器，部分功能可能不可用', 'danger');
            });
    };
    
    // 公开方法
    return {
        init,
        switchToTab,
        showNotification,
        showConfirmDialog,
        getCurrentTab: () => activeTab
    };
})();

// 在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    UIModule.init();
});

