/**
 * 论文管理模块，负责处理论文列表和详情
 */
const PapersModule = (() => {
    // 元素引用
    const papersTable = document.getElementById('papers-table');
    const papersPagination = document.getElementById('papers-pagination');
    const categoryFilters = document.getElementById('category-filters');
    const refreshPapersButton = document.getElementById('refresh-papers');
    const selectPaperTable = document.getElementById('select-paper-table');
    const confirmSelectPaperButton = document.getElementById('confirm-select-paper');
    const paperSearchInput = document.getElementById('paper-search');
    const searchPaperButton = document.getElementById('search-paper-btn');
    
    // 状态
    let papers = [];
    let currentPage = 1;
    let totalPages = 1;
    let pageSize = 10;
    let currentCategory = 'all';
    let selectedPaperId = null;
    let selectedPaperTitle = '';
    
    // 初始化
    const init = () => {
        // 加载论文类别
        loadCategories();
        
        // 加载论文列表
        loadPapers();
        
        // 绑定事件处理器
        bindEventHandlers();
    };
    
    // 绑定事件处理器
    const bindEventHandlers = () => {
        // 刷新按钮
        refreshPapersButton.addEventListener('click', () => {
            loadPapers(currentCategory, currentPage);
        });
        
        // 分类筛选事件委托
        categoryFilters.addEventListener('click', (event) => {
            if (event.target.tagName === 'A') {
                event.preventDefault();
                
                // 更新活跃状态
                document.querySelectorAll('#category-filters .dropdown-item').forEach(item => {
                    item.classList.remove('active');
                });
                event.target.classList.add('active');
                
                // 获取类别
                const category = event.target.dataset.category;
                currentCategory = category;
                
                // 重新加载论文
                loadPapers(category, 1);
            }
        });
        
        // 选择论文表格事件委托
        selectPaperTable.addEventListener('click', (event) => {
            const row = event.target.closest('tr');
            if (row) {
                // 更新选中状态
                selectPaperTable.querySelectorAll('tr').forEach(tr => {
                    tr.classList.remove('table-primary');
                });
                row.classList.add('table-primary');
                
                // 获取论文ID和标题
                selectedPaperId = row.dataset.paperId;
                selectedPaperTitle = row.dataset.paperTitle;
                
                // 启用确认按钮
                confirmSelectPaperButton.disabled = false;
            }
        });
        
        // 确认选择论文按钮
        confirmSelectPaperButton.addEventListener('click', () => {
            if (selectedPaperId) {
                // 设置当前论文
                ChatModule.setCurrentPaper(selectedPaperId, selectedPaperTitle);
                
                // 通知聊天模块
                ChatModule.addAssistantMessage(`已选择论文《${selectedPaperTitle}》，您可以开始询问该论文的相关问题。`);
                
                // 切换到聊天界面
                UIModule.switchToTab('chat');
                
                // 关闭模态框
                const modal = bootstrap.Modal.getInstance(document.getElementById('select-paper-modal'));
                modal.hide();
                
                // 重置选择状态
                resetPaperSelection();
            }
        });
        
        // 搜索论文按钮
        searchPaperButton.addEventListener('click', searchPapers);
        
        // 搜索输入框回车键
        paperSearchInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                searchPapers();
            }
        });
    };
    
    // 加载论文列表
    const loadPapers = (category = 'all', page = 1, query = '') => {
        // 更新状态
        currentPage = page;
        
        // 显示加载指示器
        papersTable.querySelector('tbody').innerHTML = '<tr><td colspan="4" class="text-center">加载中...</td></tr>';
        
        // 准备参数
        const params = {
            page: page,
            limit: pageSize
        };
        
        if (category && category !== 'all') {
            params.category = category;
        }
        
        if (query) {
            params.query = query;
        }
        
        // 发送请求
        API.getPapers(params)
            .then(response => {
                // 更新状态
                papers = response.papers;
                totalPages = Math.ceil(response.total / pageSize);
                
                // 渲染论文列表
                renderPapersList(papers);
                
                // 渲染分页
                renderPagination();
            })
            .catch(error => {
                // 显示错误消息
                papersTable.querySelector('tbody').innerHTML = `
                    <tr>
                        <td colspan="4" class="text-center text-danger">
                            加载失败: ${error.message || '未知错误'}
                        </td>
                    </tr>
                `;
            });
    };
    
    // 渲染论文列表
    const renderPapersList = (papers) => {
        const tbody = papersTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        if (papers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center">没有找到论文</td></tr>';
            return;
        }
        
        papers.forEach(paper => {
            const row = document.createElement('tr');
            row.className = 'paper-row';
            row.dataset.paperId = paper.paper_id;
            
            row.innerHTML = `
                <td>${paper.title}</td>
                <td>${paper.author || '未知作者'}</td>
                <td>
                    ${paper.category ? `<span class="badge bg-primary">${paper.category}</span>` : ''}
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-paper-btn" data-paper-id="${paper.paper_id}">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-success chat-paper-btn" data-paper-id="${paper.paper_id}" data-paper-title="${paper.title}">
                        <i class="fas fa-comments"></i>
                    </button>
                </td>
            `;
            
            // 添加查看按钮事件
            row.querySelector('.view-paper-btn').addEventListener('click', (event) => {
                event.stopPropagation();
                showPaperDetails(paper.paper_id);
            });
            
            // 添加聊天按钮事件
            row.querySelector('.chat-paper-btn').addEventListener('click', (event) => {
                event.stopPropagation();
                ChatModule.setCurrentPaper(paper.paper_id, paper.title);
                ChatModule.addAssistantMessage(`已选择论文《${paper.title}》，您可以开始询问该论文的相关问题。`);
                UIModule.switchToTab('chat');
            });
            
            // 添加行点击事件
            row.addEventListener('click', () => {
                showPaperDetails(paper.paper_id);
            });
            
            tbody.appendChild(row);
        });
    };
    
    // 渲染分页
    const renderPagination = () => {
        papersPagination.innerHTML = '';
        
        if (totalPages <= 1) {
            return;
        }
        
        // 添加上一页按钮
        const prevLi = document.createElement('li');
        prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
        prevLi.innerHTML = `<a class="page-link" href="#" data-page="${currentPage - 1}">上一页</a>`;
        papersPagination.appendChild(prevLi);
        
        // 添加页码
        for (let i = 1; i <= totalPages; i++) {
            if (
                i === 1 || 
                i === totalPages || 
                (i >= currentPage - 1 && i <= currentPage + 1)
            ) {
                const pageLi = document.createElement('li');
                pageLi.className = `page-item ${i === currentPage ? 'active' : ''}`;
                pageLi.innerHTML = `<a class="page-link" href="#" data-page="${i}">${i}</a>`;
                papersPagination.appendChild(pageLi);
            } else if (
                i === currentPage - 2 || 
                i === currentPage + 2
            ) {
                const ellipsisLi = document.createElement('li');
                ellipsisLi.className = 'page-item disabled';
                ellipsisLi.innerHTML = '<a class="page-link" href="#">...</a>';
                papersPagination.appendChild(ellipsisLi);
            }
        }
        
        // 添加下一页按钮
        const nextLi = document.createElement('li');
        nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
        nextLi.innerHTML = `<a class="page-link" href="#" data-page="${currentPage + 1}">下一页</a>`;
        papersPagination.appendChild(nextLi);
        
        // 添加分页事件
        papersPagination.querySelectorAll('.page-link').forEach(link => {
            link.addEventListener('click', function(event) {
                event.preventDefault();
                
                if (this.parentNode.classList.contains('disabled')) {
                    return;
                }
                
                const page = parseInt(this.dataset.page);
                loadPapers(currentCategory, page);
            });
        });
    };
    
    // 加载论文类别
    const loadCategories = () => {
        API.getCategories()
            .then(categories => {
                // 清空现有类别（保留"所有类别"选项）
                categoryFilters.querySelectorAll('.dropdown-item:not([data-category="all"])').forEach(item => item.remove());
                
                // 添加类别
                categories.forEach(category => {
                    const item = document.createElement('li');
                    item.innerHTML = `<a class="dropdown-item" href="#" data-category="${category}">${category}</a>`;
                    categoryFilters.appendChild(item);
                });
            })
            .catch(error => {
                console.error('加载论文类别失败:', error);
            });
    };
    
    // 显示论文详情
    const showPaperDetails = (paperId) => {
        // 显示加载指示器
        document.getElementById('modal-paper-title').textContent = '加载中...';
        document.getElementById('modal-paper-author').textContent = '';
        document.getElementById('modal-paper-category').textContent = '';
        document.getElementById('modal-paper-abstract').textContent = '加载中...';
        
        // 清空框架和实体内容
        document.getElementById('paper-framework').innerHTML = '<div class="text-center">加载中...</div>';
        document.getElementById('paper-entities-table').querySelector('tbody').innerHTML = '';
        document.getElementById('paper-relations-table').querySelector('tbody').innerHTML = '';
        
        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('paper-detail-modal'));
        modal.show();
        
        // 加载论文详情
        API.getPaperDetails(paperId)
            .then(paper => {
                // 更新基本信息
                document.getElementById('modal-paper-title').textContent = paper.title;
                document.getElementById('modal-paper-author').textContent = paper.author || '未知作者';
                document.getElementById('modal-paper-category').textContent = paper.category || '未分类';
                document.getElementById('modal-paper-filename').textContent = paper.filename || '-';
                document.getElementById('modal-paper-abstract').textContent = paper.abstract || '无摘要';
                
                // 更新框架
                renderPaperFramework(paper.framework);
                
                // 更新实体和关系
                renderPaperEntities(paper.entities);
                renderPaperRelations(paper.relations);
                
                // 设置与论文对话按钮
                document.getElementById('chat-with-modal-paper').dataset.paperId = paperId;
                document.getElementById('chat-with-modal-paper').dataset.paperTitle = paper.title;
                
                // 添加与论文对话按钮事件
                document.getElementById('chat-with-modal-paper').addEventListener('click', function() {
                    ChatModule.setCurrentPaper(this.dataset.paperId, this.dataset.paperTitle);
                    ChatModule.addAssistantMessage(`已选择论文《${this.dataset.paperTitle}》，您可以开始询问该论文的相关问题。`);
                    UIModule.switchToTab('chat');
                    modal.hide();
                });
            })
            .catch(error => {
                // 显示错误消息
                document.getElementById('modal-paper-title').textContent = '加载失败';
                document.getElementById('modal-paper-abstract').textContent = `错误: ${error.message || '未知错误'}`;
            });
    };
    
    // 渲染论文框架
    const renderPaperFramework = (framework) => {
        const frameworkContainer = document.getElementById('paper-framework');
        frameworkContainer.innerHTML = '';
        
        if (!framework || Object.keys(framework).length === 0) {
            frameworkContainer.innerHTML = '<div class="text-center">无框架数据</div>';
            return;
        }
        
        const sections = Object.keys(framework);
        
        const accordion = document.createElement('div');
        accordion.className = 'accordion';
        accordion.id = 'frameworkAccordion';
        
        sections.forEach((section, index) => {
            const content = framework[section];
            
            if (!content) {
                return;
            }
            
            const accordionItem = document.createElement('div');
            accordionItem.className = 'accordion-item';
            
            const accordionHeader = document.createElement('h2');
            accordionHeader.className = 'accordion-header';
            accordionHeader.id = `heading${index}`;
            
            const accordionButton = document.createElement('button');
            accordionButton.className = 'accordion-button collapsed';
            accordionButton.type = 'button';
            accordionButton.dataset.bsToggle = 'collapse';
            accordionButton.dataset.bsTarget = `#collapse${index}`;
            accordionButton.setAttribute('aria-expanded', 'false');
            accordionButton.setAttribute('aria-controls', `collapse${index}`);
            accordionButton.textContent = section;
            
            accordionHeader.appendChild(accordionButton);
            
            const accordionCollapse = document.createElement('div');
            accordionCollapse.id = `collapse${index}`;
            accordionCollapse.className = 'accordion-collapse collapse';
            accordionCollapse.setAttribute('aria-labelledby', `heading${index}`);
            accordionCollapse.dataset.bsParent = '#frameworkAccordion';
            
            const accordionBody = document.createElement('div');
            accordionBody.className = 'accordion-body';
            accordionBody.textContent = content;
            
            accordionCollapse.appendChild(accordionBody);
            
            accordionItem.appendChild(accordionHeader);
            accordionItem.appendChild(accordionCollapse);
            
            accordion.appendChild(accordionItem);
        });
        
        frameworkContainer.appendChild(accordion);
    };
    
    // 渲染论文实体
    const renderPaperEntities = (entities) => {
        const tbody = document.getElementById('paper-entities-table').querySelector('tbody');
        tbody.innerHTML = '';
        
        if (!entities || entities.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">无实体数据</td></tr>';
            return;
        }
        
        entities.forEach(entity => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${entity.name}</td>
                <td><span class="badge badge-${entity.type.toLowerCase()}">${entity.type}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline-info view-entity-btn" data-entity-name="${entity.name}" data-entity-type="${entity.type}">
                        <i class="fas fa-info-circle"></i>
                    </button>
                </td>
            `;
            
            // 添加查看实体按钮事件
            row.querySelector('.view-entity-btn').addEventListener('click', function() {
                showEntityDetails(this.dataset.entityName, this.dataset.entityType);
            });
            
            tbody.appendChild(row);
        });
    };
    
    // 渲染论文关系
    const renderPaperRelations = (relations) => {
        const tbody = document.getElementById('paper-relations-table').querySelector('tbody');
        tbody.innerHTML = '';
        
        if (!relations || relations.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">无关系数据</td></tr>';
            return;
        }
        
        relations.forEach(relation => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${relation.head}</td>
                <td>${relation.relation}</td>
                <td>${relation.tail}</td>
            `;
            
            tbody.appendChild(row);
        });
    };
    
    // 显示实体详情
    const showEntityDetails = (entityName, entityType) => {
        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('entity-detail-modal'));
        modal.show();
        
        // 显示加载指示器
        document.getElementById('entity-name').textContent = '加载中...';
        document.getElementById('entity-type').textContent = '';
        document.getElementById('entity-relations-list').innerHTML = '<li>加载中...</li>';
        document.getElementById('entity-papers-list').innerHTML = '<li>加载中...</li>';
        
        // 加载实体详情
        API.getEntityDetails(entityName, entityType)
            .then(entity => {
                // 更新基本信息
                document.getElementById('entity-name').textContent = entity.name;
                document.getElementById('entity-type').textContent = entity.type || '';
                
                // 更新关系列表
                renderEntityRelations(entity.relations);
                
                // 更新论文列表
                renderEntityPapers(entity.papers);
                
                // 设置询问实体按钮
                document.getElementById('ask-about-entity').dataset.entityName = entity.name;
                
                // 添加询问实体按钮事件
                document.getElementById('ask-about-entity').addEventListener('click', function() {
                    const questionText = `请告诉我关于${this.dataset.entityName}的信息`;
                    
                    // 关闭模态框
                    modal.hide();
                    
                    // 切换到聊天界面
                    UIModule.switchToTab('chat');
                    
                    // 添加问题
                    ChatModule.addUserMessage(questionText);
                    
                    // 发送请求
                    API.askQuestion(questionText)
                        .then(response => {
                            ChatModule.addAssistantMessage(response.answer);
                        })
                        .catch(error => {
                            ChatModule.addAssistantMessage(`抱歉，发生了错误：${error.message || '未知错误'}`);
                        });
                });
            })
            .catch(error => {
                // 显示错误消息
                document.getElementById('entity-name').textContent = '加载失败';
                document.getElementById('entity-relations-list').innerHTML = `<li class="text-danger">错误: ${error.message || '未知错误'}</li>`;
                document.getElementById('entity-papers-list').innerHTML = '';
            });
    };
    
    // 渲染实体关系
    const renderEntityRelations = (relations) => {
        const listElement = document.getElementById('entity-relations-list');
        listElement.innerHTML = '';
        
        if (!relations || relations.length === 0) {
            listElement.innerHTML = '<li class="text-muted">无相关关系</li>';
            return;
        }
        
        relations.forEach(relation => {
            const item = document.createElement('li');
            
            if (relation.direction === 'outgoing') {
                item.innerHTML = `
                    <strong>${relation.relation}</strong> → 
                    <span class="entity-link" data-entity-name="${relation.entity}" data-entity-type="${relation.entity_type}">
                        ${relation.entity}
                    </span>
                `;
            } else {
                item.innerHTML = `
                    <span class="entity-link" data-entity-name="${relation.entity}" data-entity-type="${relation.entity_type}">
                        ${relation.entity}
                    </span>
                    → <strong>${relation.relation}</strong>
                `;
            }
            
            // 添加实体链接点击事件
            item.querySelector('.entity-link').addEventListener('click', function() {
                showEntityDetails(this.dataset.entityName, this.dataset.entityType);
            });
            
            listElement.appendChild(item);
        });
    };
    
    // 渲染实体相关论文
    const renderEntityPapers = (papers) => {
        const listElement = document.getElementById('entity-papers-list');
        listElement.innerHTML = '';
        
        if (!papers || papers.length === 0) {
            listElement.innerHTML = '<li class="text-muted">无相关论文</li>';
            return;
        }
        
        papers.forEach(paper => {
            const item = document.createElement('li');
            
            item.innerHTML = `
                <span class="paper-link" data-paper-id="${paper.paper_id}">
                    ${paper.title}
                </span>
                ${paper.category ? `<span class="badge bg-primary">${paper.category}</span>` : ''}
            `;
            
            // 添加论文链接点击事件
            item.querySelector('.paper-link').addEventListener('click', function() {
                showPaperDetails(this.dataset.paperId);
            });
            
            listElement.appendChild(item);
        });
    };
    
    // 搜索论文
    const searchPapers = () => {
        const query = paperSearchInput.value.trim();
        
        if (query) {
            loadPapers(currentCategory, 1, query);
        } else {
            loadPapers(currentCategory, 1);
        }
    };
    
    // 重置论文选择
    const resetPaperSelection = () => {
        selectedPaperId = null;
        selectedPaperTitle = '';
        confirmSelectPaperButton.disabled = true;
        
        selectPaperTable.querySelectorAll('tr').forEach(tr => {
            tr.classList.remove('table-primary');
        });
    };
    
    // 加载选择论文表格
    const loadSelectPaperTable = () => {
        // 显示加载指示器
        selectPaperTable.querySelector('tbody').innerHTML = '<tr><td colspan="3" class="text-center">加载中...</td></tr>';
        
        // 发送请求
        API.getPapers({ limit: 20 })
            .then(response => {
                const papers = response.papers;
                
                // 渲染表格
                renderSelectPaperTable(papers);
            })
            .catch(error => {
                // 显示错误消息
                selectPaperTable.querySelector('tbody').innerHTML = `
                    <tr>
                        <td colspan="3" class="text-center text-danger">
                            加载失败: ${error.message || '未知错误'}
                        </td>
                    </tr>
                `;
            });
    };
    
    // 渲染选择论文表格
    const renderSelectPaperTable = (papers) => {
        const tbody = selectPaperTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        if (papers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center">没有找到论文</td></tr>';
            return;
        }
        
        papers.forEach(paper => {
            const row = document.createElement('tr');
            row.dataset.paperId = paper.paper_id;
            row.dataset.paperTitle = paper.title;
            
            row.innerHTML = `
                <td>${paper.title}</td>
                <td>${paper.author || '未知作者'}</td>
                <td>${paper.category || ''}</td>
            `;
            
            tbody.appendChild(row);
        });
    };
    
    // 公开方法
    return {
        init,
        loadSelectPaperTable,
        showPaperDetails,
        showEntityDetails
    };
})();

// 在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    PapersModule.init();
});

