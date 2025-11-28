**[English Version](README.md)**

<h1 align="center">👕 file1.agent: AI智能体的文件管理工具</h1>

![file1.agent](fig.png)
<h2 align="center">  告别main_fixed.py：用file1.agent清理AI智能体留下的烂摊子</h2>

还在被满屏的 main_fixed.py、file_improved.py 搞得头大吗？
file1.agent 帮你从根源上解决 AI agent 代码越写越乱的问题。它能智能**识别并清理重复文件、临时文件和各种伪造数据**，让你的项目目录一下子清爽起来。file1.agent 还会自动构建文件关系图，把文件、数据和图片的关系直观呈现出来，复杂项目也能一眼看懂。

借助对文本、图像、PDF 等多模态内容的分析能力，file1.agent 可以真正理解你整个工作区的上下文。而 file1.agent 自动生成的文件关系图和文件摘要，还能直接**作为 agent 的长期记忆**（agent memory），让 AI agent 对你的项目保持持续、结构化的理解。

## 功能特性

- **文件摘要**：自动为文件和目录生成摘要
- **重复文件检测**：使用基于LLM的比较识别重复文件
- **模拟数据检测**：检测并删除模拟/测试数据文件
- **关系可视化**：创建显示文件关系的可视化图表
- **视觉模型集成**：从图像和PDF中提取并分析内容
- **重排序支持**：使用重排序模型提高相关性评分
