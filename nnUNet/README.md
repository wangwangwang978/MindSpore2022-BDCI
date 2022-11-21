
├──nnUNet
    ├── kits19                                // kits19原始的数据集结构
    ├── src
    │   ├──nnunet
    │   │  ├──evaluation                    // 验证精度指标
    │   │  ├──experiment_planning           // 网络执行计划生成
    │   │  ├──inference                     // 推理
    │   │  ├──network_architecture          // 基本网络结构
    │   │  ├──postprocessing                // 网络结果分析
    │   │  ├──preprocessing                 // 网络数据准备变换
    │   │  ├──run                           // 运行加载文件
    │   │  ├──training                      // 训练网络结构
    │   │  ├──utilities                     // 常用工具
    │   │  ├──configuration.py              // 配置函数
    │   │  ├──generate_testset.py           // 交叉验证数据集提取
    │   │  ├──paths.py                      // 路径导入设置
    │   ├──nnUNetFrame                      // 存放数据集
    ├── BDCI.ipynb                            // 代码流程
    ├── requirements.txt                       // 第三方库文件
    ├── train.py                              // 训练脚本
    ├── eval.py                             // 评估脚本
