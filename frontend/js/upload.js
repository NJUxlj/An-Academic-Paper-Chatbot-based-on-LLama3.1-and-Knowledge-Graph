
/**
 * 上传模块，负责处理论文上传和分析
 */
const UploadModule = (() => {
    // 元素引用
    const uploadForm = document.getElementById('upload-form');
    const pdfFileInput = document.getElementById('pdf-file');
    const uploadButton = document.getElementById('upload-button');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = uploadProgress.querySelector('.progress-bar');
    const uploadStatus = document.getElementById('upload-status');
    const analysisResults = document.getElementById('analysis-results');
    const chatWithPaperButton = document.getElementById('chat-with-paper');
    
    // 上传状态
    let isUploading = false;
    let uploadedPaperId = null;
    let uploadedPaperTitle = '';
    
    // 初始化
    const init = () => {
        // 绑定事件处理器
        bindEventHandlers();
    };
    
    // 绑定事件处理器
    const bindEventHandlers = () => {
        // 表单提交
        uploadForm.addEventListener('submit', (event) => {
            event.preventDefault();
            uploadPaper();
        });
        
        // 与论文对话按钮
        chatWithPaperButton.addEventListener('click', () => {
            if (uploadedPaperId && uploadedPaperTitle) {
                // 设置当前论文
                ChatModule.setCurrentPaper(uploadedPaperId, uploadedPaperTitle);
                
                // 通知聊天模块
                ChatModule.addAssistantMessage(`已选择论文《${uploadedPaperTitle}》，您可以开始询问该论文的相关问题。`);
                
                // 切换到聊天界面
                UIModule.switchToTab('chat');
            }
        });
    };
    
    // 上传论文
    const uploadPaper = () => {
        // 获取文件
        const file = pdfFileInput.files[0];
        
        if (!file) {
            showStatus('请选择PDF文件', 'danger');
            return;
        }
        
        // 检查文件类型
        if (file.type !== 'application/pdf') {
            showStatus('只支持PDF文件', 'danger');
            return;
        }
        
        // 检查文件大小
        if (file.size > 20 * 1024 * 1024) {
            showStatus('文件大小不能超过20MB', 'danger');
            return;
        }
        
        // 更新UI
        isUploading = true;
        uploadButton.disabled = true;
        uploadProgress.classList.remove('d-none');
        progressBar.style.width = '0%';
        showStatus('正在上传文件...', 'info');
        
        // 上传文件
        API.uploadFile(file, (progress) => {
            progressBar.style.width = `${progress}%`;
        })
            .then(response => {
                // 更新进度条
                progressBar.style.width = '100%';
                
                // 保存上传结果
                uploadedPaperId = response.paper_id;
                uploadedPaperTitle = response.title;
                
                // 显示成功消息
                showStatus(`文件上传成功！论文《${response.title}》已处理`, 'success');
                
                // 显示分析结果
                displayAnalysisResults(response);
                
                // 重置上传状态
                isUploading = false;
                uploadButton.disabled = false;
            })
            .catch(error => {
                // 更新进度条
                progressBar.style.width = '100%';
                progressBar.classList.remove('bg-primary');
                progressBar.classList.add('bg-danger');
                
                // 显示错误消息
                showStatus(`上传失败: ${error.message || '未知错误'}`, 'danger');
                
                // 重置上传状态
                isUploading = false;
                uploadButton.disabled = false;
            });
    };
    
    // 显示状态消息
    const showStatus = (message, type = 'info') => {
        uploadStatus.textContent = message;
        uploadStatus.className = `alert alert-${type}`;
        uploadStatus.classList.remove('d-none');
    };
    
    // 显示分析结果
    const displayAnalysisResults = (paper) => {
        // 显示分析结果卡片
        analysisResults.classList.remove('d-none');
        
        // 更新论文信息
        document.getElementById('paper-title').textContent = paper.title || '-';
        document.getElementById('paper-author').textContent = paper.author || '-';
        document.getElementById('paper-category').textContent = paper.category || '-';
        document.getElementById('paper-abstract').textContent = paper.abstract || '无摘要';
        
        // 更新处理统计
        document.getElementById('chunk-count').textContent = paper.statistics?.chunks || '0';
        document.getElementById('entity-count').textContent = paper.statistics?.entities || '0';
        document.getElementById('relation-count').textContent = paper.statistics?.relations || '0';
        
        // 更新类别置信度
        displayCategoryConfidence(paper.category_confidence);
    };
    
    // 显示类别置信度
    const displayCategoryConfidence = (confidence) => {
        const container = document.getElementById('category-confidence');
        container.innerHTML = '';
        
        if (!confidence || Object.keys(confidence).length === 0) {
            container.innerHTML = '<p class="text-muted">无类别置信度数据</p>';
            return;
        }
        
        // 按置信度排序
        const sortedCategories = Object.entries(confidence)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);  // 只显示前5个
        
        sortedCategories.forEach(([category, score]) => {
            const percentage = Math.round(score * 100);
            
            const progressDiv = document.createElement('div');
            progressDiv.className = 'mb-2';
            
            progressDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span>${category}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar ${percentage > 70 ? 'bg-success' : 'bg-primary'}" 
                        role="progressbar" 
                        style="width: ${percentage}%" 
                        aria-valuenow="${percentage}" 
                        aria-valuemin="0" 
                        aria-valuemax="100">
                    </div>
                </div>
            `;
            
            container.appendChild(progressDiv);
        });
    };
    
    // 重置上传表单
    const resetUploadForm = () => {
        uploadForm.reset();
        uploadProgress.classList.add('d-none');
        uploadStatus.classList.add('d-none');
        analysisResults.classList.add('d-none');
        
        progressBar.style.width = '0%';
        progressBar.classList.remove('bg-danger');
        progressBar.classList.add('bg-primary');
        
        uploadedPaperId = null;
        uploadedPaperTitle = '';
    };
    
    // 公开方法
    return {
        init,
        resetUploadForm
    };
})();

// 在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    UploadModule.init();
});

