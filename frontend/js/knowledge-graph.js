/**
 * 知识图谱可视化模块，负责处理知识图谱显示和交互
 */
const KnowledgeGraphModule = (() => {
    // 元素引用
    const kgVisualization = document.getElementById('kg-visualization');
    const kgStats = document.getElementById('kg-stats');
    const entityTypeFilters = document.getElementById('entity-type-filters');
    const kgRefreshButton = document.getElementById('kg-refresh');
    const entitySearchInput = document.getElementById('entity-search');
    const searchEntityButton = document.getElementById('search-entity-btn');
    
    // 图谱状态
    let graph = null;
    let graphData = {
        nodes: [],
        links: []
    };
    let currentFilter = 'all';
    let highlightedNodes = new Set();
    
    // 颜色映射
    const colorMap = {
        MODEL: '#4285f4',
        METHOD: '#34a853',
        DATASET: '#ea4335',
        METRIC: '#fbbc05',
        TASK: '#9c27b0',
        FRAMEWORK: '#673ab7',
        Paper: '#3f51b5',
        default: '#607d8b'
    };
    
    // 初始化
    const init = () => {
        // 加载实体类型
        loadEntityTypes();
        
        // 初始化知识图谱
        initGraph();
        
        // 加载知识图谱数据
        loadKnowledgeGraph();
        
        // 加载知识图谱统计信息
        loadStatistics();
        
        // 绑定事件处理器
        bindEventHandlers();
    };
    
    // 绑定事件处理器
    const bindEventHandlers = () => {
        // 刷新按钮
        kgRefreshButton.addEventListener('click', () => {
            loadKnowledgeGraph(currentFilter);
            loadStatistics();
        });
        
        // 实体类型筛选事件委托
        entityTypeFilters.addEventListener('click', (event) => {
            if (event.target.tagName === 'A') {
                event.preventDefault();
                
                // 更新活跃状态
                document.querySelectorAll('#entity-type-filters .dropdown-item').forEach(item => {
                    item.classList.remove('active');
                });
                event.target.classList.add('active');
                
                // 获取类型
                const entityType = event.target.dataset.type;
                currentFilter = entityType;
                
                // 重新加载知识图谱
                loadKnowledgeGraph(entityType);
            }
        });
        
        // 搜索实体按钮
        searchEntityButton.addEventListener('click', searchEntity);
        
        // 搜索输入框回车键
        entitySearchInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                searchEntity();
            }
        });
    };
    
    // 初始化图谱
    const initGraph = () => {
        // 创建ForceGraph实例
        graph = ForceGraph()(kgVisualization)
            .nodeId('id')
            .nodeLabel('name')
            .nodeColor(node => {
                if (highlightedNodes.has(node.id)) {
                    return '#ff0000';  // 高亮节点为红色
                }
                return node.color || colorMap.default;
            })
            .nodeVal(node => {
                // 根据连接数调整节点大小
                const links = graphData.links.filter(link => 
                    link.source === node.id || link.target === node.id
                );
                return 5 + Math.min(10, links.length);
            })
            .linkSource('source')
            .linkTarget('target')
            .linkLabel('label')
            .linkDirectionalArrowLength(6)
            .linkDirectionalArrowRelPos(1)
            .linkCurvature(0.2)
            .linkWidth(2)
            .onNodeHover(node => {
                // 节点悬停效果
                kgVisualization.style.cursor = node ? 'pointer' : null;
            })
            .onNodeClick(node => {
                // 节点点击事件
                PapersModule.showEntityDetails(node.name, node.type);
            })
            .d3Force('charge', d3.forceManyBody().strength(-100))
            .d3Force('center', d3.forceCenter(kgVisualization.clientWidth / 2, kgVisualization.clientHeight / 2))
            .d3Force('link', d3.forceLink().id(d => d.id).distance(80))
            .cooldownTicks(100)
            .onEngineStop(() => graph.zoomToFit(400, 20))
            .graphData(graphData);
    };
    
    // 加载知识图谱数据
    const loadKnowledgeGraph = (entityType = 'all') => {
        // 显示加载指示器
        kgVisualization.innerHTML = '<div class="text-center mt-5">加载知识图谱数据...</div>';
        
        // 准备参数
        const params = {};
        if (entityType && entityType !== 'all') {
            params.entity_type = entityType;
        }
        
        // 发送请求
        API.getKnowledgeGraphData(params)
            .then(data => {
                // 更新图谱数据
                graphData = processGraphData(data);
                
                // 更新图谱
                updateGraph();
            })
            .catch(error => {
                // 显示错误消息
                kgVisualization.innerHTML = `
                    <div class="text-center mt-5 text-danger">
                        加载失败: ${error.message || '未知错误'}
                    </div>
                `;
            });
    };
    
    // 处理图谱数据
    const processGraphData = (data) => {
        // 处理节点
        const nodes = data.nodes.map(node => {
            const type = node.labels[0];
            return {
                id: node.id,
                name: node.properties.name || `节点${node.id}`,
                type: type,
                color: colorMap[type] || colorMap.default,
                properties: node.properties
            };
        });
        
        // 处理连接
        const links = data.relationships.map(rel => {
            return {
                source: rel.source,
                target: rel.target,
                type: rel.type,
                label: rel.type,
                properties: rel.properties
            };
        });
        
        return { nodes, links };
    };
    
    // 更新图谱
    const updateGraph = () => {
        // 清除高亮节点
        highlightedNodes.clear();
        
        // 更新图谱数据
        graph.graphData(graphData);
        
        // 自动调整视图
        setTimeout(() => {
            graph.zoomToFit(400, 20);
        }, 1000);
    };
    
    // 加载实体类型
    const loadEntityTypes = () => {
        API.getKnowledgeGraphStats()
            .then(stats => {
                // 清空现有类型（保留"所有类型"选项）
                entityTypeFilters.querySelectorAll('.dropdown-item:not([data-type="all"])').forEach(item => item.remove());
                
                // 获取实体类型
                const entityTypes = Object.keys(stats.entity_counts || {});
                
                // 添加类型
                entityTypes.forEach(type => {
                    const count = stats.entity_counts[type];
                    
                    const item = document.createElement('li');
                    item.innerHTML = `
                        <a class="dropdown-item" href="#" data-type="${type}">
                            ${type} <span class="badge bg-secondary rounded-pill">${count}</span>
                        </a>
                    `;
                    entityTypeFilters.appendChild(item);
                });
            })
            .catch(error => {
                console.error('加载实体类型失败:', error);
            });
    };
    
    // 加载知识图谱统计信息
    const loadStatistics = () => {
        API.getKnowledgeGraphStats()
            .then(stats => {
                // 渲染统计信息
                renderStatistics(stats);
            })
            .catch(error => {
                console.error('加载知识图谱统计信息失败:', error);
                
                // 显示错误消息
                kgStats.innerHTML = `
                    <div class="text-center text-danger">
                        加载统计信息失败: ${error.message || '未知错误'}
                    </div>
                `;
            });
    };
    
    // 渲染统计信息
    const renderStatistics = (stats) => {
        kgStats.innerHTML = '';
        
        if (!stats) {
            kgStats.innerHTML = '<div class="text-center text-muted">无统计信息</div>';
            return;
        }
        
        // 实体总数
        let totalEntities = 0;
        for (const type in stats.entity_counts) {
            totalEntities += stats.entity_counts[type];
        }
        
        // 关系总数
        let totalRelations = 0;
        for (const type in stats.relation_counts) {
            totalRelations += stats.relation_counts[type];
        }
        
        // 创建统计项
        const statsItems = [
            { label: '实体数量', value: totalEntities },
            { label: '关系数量', value: totalRelations },
            { label: '论文数量', value: stats.paper_count || 0 },
            { label: '三元组数量', value: stats.total_triples || 0 }
        ];
        
        statsItems.forEach(item => {
            const div = document.createElement('div');
            div.className = 'kg-stats-item';
            div.innerHTML = `
                <h6>${item.label}</h6>
                <p>${item.value.toLocaleString()}</p>
            `;
            kgStats.appendChild(div);
        });
    };
    
    // 搜索实体
    const searchEntity = () => {
        const query = entitySearchInput.value.trim();
        
        if (!query) {
            return;
        }
        
        // 清除高亮节点
        highlightedNodes.clear();
        
        // 搜索实体
        API.searchEntities(query, currentFilter === 'all' ? null : currentFilter)
            .then(entities => {
                if (entities.length === 0) {
                    alert('没有找到匹配的实体');
                    return;
                }
                
                // 高亮匹配的节点
                highlightMatchingNodes(entities);
            })
            .catch(error => {
                alert(`搜索失败: ${error.message || '未知错误'}`);
            });
    };
    
    // 高亮匹配的节点
    const highlightMatchingNodes = (entities) => {
        // 找到匹配的节点
        const matchingNodes = graphData.nodes.filter(node => 
            entities.some(entity => 
                entity.name === node.name && 
                (!entity.type || entity.type === node.type)
            )
        );
        
        if (matchingNodes.length === 0) {
            alert('在当前图谱中没有找到匹配的实体');
            return;
        }
        
        // 高亮节点
        matchingNodes.forEach(node => {
            highlightedNodes.add(node.id);
        });
        
        // 将第一个匹配节点居中显示
        const firstNode = matchingNodes[0];
        if (firstNode) {
            graph.centerAt(firstNode.x, firstNode.y, 1000);
            graph.zoom(2, 1000);
        }
        
        // 更新图谱
        graph.refresh();
    };
    
    // 重置高亮
    const resetHighlight = () => {
        highlightedNodes.clear();
        graph.refresh();
    };
    
    // 获取图谱节点
    const getGraphNode = (entityName, entityType) => {
        return graphData.nodes.find(node => 
            node.name === entityName && 
            (!entityType || node.type === entityType)
        );
    };
    
    // 聚焦于实体
    const focusOnEntity = (entityName, entityType) => {
        // 查找节点
        const node = getGraphNode(entityName, entityType);
        
        if (!node) {
            console.warn(`实体 ${entityName} 不在当前图谱中`);
            return false;
        }
        
        // 高亮节点
        highlightedNodes.clear();
        highlightedNodes.add(node.id);
        
        // 居中并放大
        graph.centerAt(node.x, node.y, 1000);
        graph.zoom(2, 1000);
        
        // 更新图谱
        graph.refresh();
        
        return true;
    };
    
    // 提取邻居节点
    const extractNeighbors = (entityName, entityType, depth = 1) => {
        // 查找中心节点
        const centralNode = getGraphNode(entityName, entityType);
        
        if (!centralNode) {
            return { nodes: [], links: [] };
        }
        
        // 收集要显示的节点ID
        const nodeIds = new Set([centralNode.id]);
        
        // 通过BFS收集邻居节点
        let frontier = [centralNode.id];
        let currentDepth = 0;
        
        while (currentDepth < depth) {
            const nextFrontier = [];
            
            for (const nodeId of frontier) {
                // 查找连接到该节点的边
                graphData.links.forEach(link => {
                    if (link.source === nodeId && !nodeIds.has(link.target)) {
                        nodeIds.add(link.target);
                        nextFrontier.push(link.target);
                    } else if (link.target === nodeId && !nodeIds.has(link.source)) {
                        nodeIds.add(link.source);
                        nextFrontier.push(link.source);
                    }
                });
            }
            
            frontier = nextFrontier;
            currentDepth++;
            
            if (frontier.length === 0) {
                break;
            }
        }
        
        // 过滤节点和边
        const nodes = graphData.nodes.filter(node => nodeIds.has(node.id));
        const links = graphData.links.filter(link => 
            nodeIds.has(link.source) && nodeIds.has(link.target)
        );
        
        return { nodes, links };
    };
    
    // 将邻居子图加载到视图中
    const loadNeighborhood = (entityName, entityType, depth = 1) => {
        const subgraph = extractNeighbors(entityName, entityType, depth);
        
        if (subgraph.nodes.length === 0) {
            alert(`未找到实体 ${entityName} 或其邻居`);
            return;
        }
        
        // 更新图谱数据
        graphData = subgraph;
        
        // 清除高亮
        highlightedNodes.clear();
        
        // 更新图谱
        updateGraph();
        
        // 高亮中心节点
        const centralNode = getGraphNode(entityName, entityType);
        if (centralNode) {
            highlightedNodes.add(centralNode.id);
            graph.refresh();
        }
    };
    
    // 下载图谱数据
    const downloadGraphData = () => {
        // 准备下载数据
        const downloadData = {
            timestamp: new Date().toISOString(),
            filter: currentFilter,
            data: graphData
        };
        
        // 转换为JSON字符串
        const jsonString = JSON.stringify(downloadData, null, 2);
        
        // 创建Blob
        const blob = new Blob([jsonString], { type: 'application/json' });
        
        // 创建下载链接
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `knowledge_graph_${new Date().getTime()}.json`;
        
        // 触发下载
        document.body.appendChild(a);
        a.click();
        
        // 清理
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
    
    // 公开方法
    return {
        init,
        loadKnowledgeGraph,
        focusOnEntity,
        loadNeighborhood,
        resetHighlight,
        downloadGraphData
    };
})();

// 在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    KnowledgeGraphModule.init();
});

