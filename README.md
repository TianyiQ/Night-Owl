这份文档需动态更新。
TODO 表示还没来得及写文档的部分。

## 算法思路

- 目前做法：先对测试数据做低光照提亮，然后用 DSFD 识别。DSFD 采用腾讯在 WIDERFACE 上预训练好的参数。
    - 分数：0.373434
- 进一步的思路：（待讨论）
    - LLIE 换成 Zero-DCE ？
    - 在 DARKFACE 训练集上 finetune ？（训练样本先经过提亮？）
        - 换用课上提到的这个（支持训练的）框架？ https://github.com/yxlijun/DSFD.pytorch
    - 对 DARKFACE 测试集进行无监督学习作为辅助？（这么干符合规则吗？）
    - ……（待讨论）

## 目录结构

- 整体基于 DSFD 修改，整个框架和 DSFD 基本相同。
    - 不少代码文件进行了改动，并新增了部分代码文件。
        - 所有有关 DARKFACE 的代码均为新增，大部分修改自 DSFD 中已有的 WIDERFACE 代码。
    - 经过修改，整个代码已经可以在高版本 torch 下运行，无需使用 DSFD 的原始 torch 版本。
- `./data/` ：用来读取数据集的代码。
    - 注意，这个文件夹里只有处理代码，不存放数据集。
    - 目前只用到了其中的 `darkface.py` 。
- `./enhance/` ：新增的用于低光照提亮的代码。
    - 基于 https://github.com/pvnieo/Low-light-Image-Enhancement/ （以下简称 LLIE ）修改而得。
        - 这个方法是非深度学习的传统方法。之后可以考虑换成 Zero-DCE 。
- `./layers/` ：来自 DSFD ，TODO
- `./model/` ：来自 DSFD ，TODO
- `./utils/` ：来自 DSFD ，TODO
- `./weights/` ：存放模型参数（ `.pth` 文件）
- `./demo.py` ：单张图片 inference 代码，不带低光照提亮。
- `./darkface_val.py` ：针对 DARKFACE 的多张图片批量 inference 代码，带低光照提亮。
- `./face_ssd.py` `./fddb_test.py` ：来自 DSFD ，TODO

## 使用方式

- `.pth` 文件过大，故不在 github 上存放。
    - 在 `./weights/PLACEHOLDER_FOR_PTH_FILES.txt` 中列出了所有 `.pth` 文件的名字。
    - `.pth` 文件在 https://boya.ai.pku.edu.cn/openai/#/modelDev/algorithmManager 中下载（名字以“【模型参数】”开头的那些算法）。
- pku 算力平台使用的注意事项：从上传的算法建立 notebook 后，在 notebook 中对 `/code/` 所做的更改 **会** 被更新到算法里去。所以建议每次建立 `notebook` 之前都重新上传一下要用的算法。
- 平台上用的镜像，是 “我的镜像” 中的 mirror:ini
    - 装了 conda ，各种包（包括合适版本的 cudatoolkit ）都装在 conda 的 PyEnv 这个环境中（不是 base ！）
    - 由于未知原因， conda 命令无法保持在 PATH 中，所以要进入 PyEnv 需要依次进行以下指令（而不是 `conda activate PyEnv` ）：
        - `/root/anaconda3/bin/conda init`
        - `. ~/.bashrc`
        - `conda activate PyEnv`
- 算力平台上，用 DARKFACE 测试集进行测试的代码（可以直接填入训练任务的运行命令一栏）：
    - 如果是本地测试，前四行可以忽略，第五行不变。
```bash
/root/anaconda3/bin/conda init
. ~/.bashrc
conda activate PyEnv
cd /code
python -W ignore::UserWarning ./darkface_val.py
```

## `./data/darkface.py`

- 用于从 `/dataset/image/` 中读取 DARKFACE 的输入图片（训练集或测试集），从 `/dataset/label/` 中读取 DARKFACE 的标注（训练集）。

## `./darkface_val.py`

- 使用 `./data/darkface.py` 读取 DARKFACE 测试集，然后进行测试。
- `.txt` 格式的 inference 结果会输出在 `/model/txts/` 中（注意是 `/model` 不是 `./model` ），格式与 DARKFACE 网站提交格式相同。
- `.png` 格式的可视化结果会输出在 `/model/pics/` 中。这会大幅增加输出文件的大小，故默认不启用。把代码中 `vis_detections(img_id.split('.')[0], image, dets, H, W)` 一句取消注释，即可启用可视化。
- 3080 卡上，大约每张图 11s 。

## `./enhance/`

- 基于 LLIE 修改而得。
- 只用到了 `exposure_enhancement.py` 和 `utils.py` 两个文件。
- 在时间开销的瓶颈处添加了一个 4x4 倍降采样，单张图提亮用时从 40s 降到 1s 。（单张图 DSFD 的推理用时约为 11s ）
    - 如果是 2x2 倍降采样，则单张图提亮用时约 7s 。这样做虽然视觉效果好于 4x4 ，但最终的 mAP score 似乎没有优势。
- 尝试用 CuPy 来用 GPU 加速，但发现速度没啥改进，而且会爆显存。