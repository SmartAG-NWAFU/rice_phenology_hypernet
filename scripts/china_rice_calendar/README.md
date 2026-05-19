# ChinaRiceCalendar 数据下载

从 Harvard Dataverse 下载中国水稻物候日历 GeoTIFF 数据。

## 数据来源

- **数据集**: ChinaRiceCalendar
- **DOI**: [10.7910/DVN/EUP8EY](https://doi.org/10.7910/DVN/EUP8EY)
- **版本**: 8
- **文件类型**: rice_pixels (像元级数据)

## 下载文件

共 9 个 GeoTIFF 文件（三季 × 三阶段）：

| 季别 | 阶段 | 文件名 |
|------|------|--------|
| Early | transplanting | `Early_rice_transplanting_dates_2003_2022_rice_pixels.tif` |
| Early | heading | `Early_rice_heading_dates_2003_2022_rice_pixels.tif` |
| Early | maturity | `Early_rice_maturity_dates_2003_2022_rice_pixels.tif` |
| Middle | transplanting | `Middle_rice_transplanting_dates_2003_2022_rice_pixels.tif` |
| Middle | heading | `Middle_rice_heading_dates_2003_2022_rice_pixels.tif` |
| Middle | maturity | `Middle_rice_maturity_dates_2003_2022_rice_pixels.tif` |
| Late | transplanting | `Late_rice_transplanting_dates_2003_2022_rice_pixels.tif` |
| Late | heading | `Late_rice_heading_dates_2003_2022_rice_pixels.tif` |
| Late | maturity | `Late_rice_maturity_dates_2003_2022_rice_pixels.tif` |

时间窗口: 2003-2022

## 使用方法

### 预览下载内容（不实际下载）

```bash
python scripts/china_rice_calendar/download_rice_calendar.py --dry-run
```

### 下载数据

```bash
python scripts/china_rice_calendar/download_rice_calendar.py
```

### 指定输出目录

```bash
python scripts/china_rice_calendar/download_rice_calendar.py --output-dir /path/to/output
```

## 输出

默认输出路径:
```
data/external/china_rice_calendar/dataverse_v8/rice_pixels/2003_2022/
├── Early_rice_transplanting_dates_2003_2022_rice_pixels.tif
├── Early_rice_heading_dates_2003_2022_rice_pixels.tif
├── Early_rice_maturity_dates_2003_2022_rice_pixels.tif
├── Middle_rice_transplanting_dates_2003_2022_rice_pixels.tif
├── Middle_rice_heading_dates_2003_2022_rice_pixels.tif
├── Middle_rice_maturity_dates_2003_2022_rice_pixels.tif
├── Late_rice_transplanting_dates_2003_2022_rice_pixels.tif
├── Late_rice_heading_dates_2003_2022_rice_pixels.tif
├── Late_rice_maturity_dates_2003_2022_rice_pixels.tif
└── download_manifest.json
```

## 验证

下载完成后，`download_manifest.json` 包含：
- 数据集 DOI
- Dataverse 版本号
- 下载时间 (UTC)
- 每个文件的：
  - 文件名
  - Dataverse file ID
  - 文件大小 (bytes)
  - MD5 校验和
  - 本地路径

可使用以下命令验证文件完整性：

```bash
# 检查文件数量
ls -la data/external/china_rice_calendar/dataverse_v8/rice_pixels/2003_2022/*.tif | wc -l
# 应输出: 9

# 检查文件大小与 manifest 是否一致
python -c "
import json
from pathlib import Path
manifest = json.loads(Path('data/external/china_rice_calendar/dataverse_v8/rice_pixels/2003_2022/download_manifest.json').read_text())
for f in manifest['files']:
    actual = Path(f['local_path']).stat().st_size
    expected = f['size_bytes']
    status = 'OK' if actual == expected else 'MISMATCH'
    print(f'{status}: {f[\"filename\"]} ({actual} vs {expected})')
"
```

## 注意事项

- 数据目录 `data/external/` 已被 `.gitignore` 忽略，不会被提交到版本控制
- 若本地已有同名文件且大小匹配，将跳过下载
- 下载使用原子写入（先写临时文件后重命名），避免中断导致文件损坏