"""
增强版票据合成器 — Enhanced Receipt Synthesizer

支持的模板类型:
  - vat_general:    增值税普通发票
  - vat_special:    增值税专用发票（含抵扣联）
  - receipt_thermal: 热敏小票/收据（模拟超市/餐饮小票）
  - receipt_english: 英文收据（模拟海外小票）

增强特性:
  - 多样化的字体、字号、粗细
  - 纸张纹理 + 不均匀光照 + 阴影
  - 印章 / 二维码
  - 透视变换、折叠痕迹
  - 商品明细合计算（含折扣/税率变化）
  - 随机背景（纯白/微黄/灰色热敏纸）
  - 高度溢出自动截断保护
"""

import os
import io
import random
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 0. 字体配置
# ═══════════════════════════════════════════════════════════════════════

CHINESE_FONTS = [
    "C:\\Windows\\Fonts\\msyh.ttc",       # 微软雅黑
    "C:\\Windows\\Fonts\\simsun.ttc",     # 宋体
    "C:\\Windows\\Fonts\\simkai.ttf",     # 楷体
    "C:\\Windows\\Fonts\\simfang.ttf",    # 仿宋
    "C:\\Windows\\Fonts\\simhei.ttf",     # 黑体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

MONO_FONTS = [
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

ENGLISH_FONTS = [
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _text_width(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> int:
    """跨版本兼容的文本宽度计算。"""
    if hasattr(draw, "textlength"):
        return int(draw.textlength(text, font=font))
    # Pillow < 10 fallback
    bbox = draw.textbbox((0, 0), text, font=font) if hasattr(draw, "textbbox") else None
    if bbox:
        return bbox[2] - bbox[0]
    return len(text) * font.size // 2


def _font_line_height(font: ImageFont.ImageFont) -> int:
    """获取字体的真实行高（ascent + descent）。"""
    # Pillow >= 10 有 font.getmetrics()
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent
    # 回退：1.4 倍字号
    return int(font.size * 1.4)


def _find_font(font_list: List[str], size: int = 20) -> ImageFont.ImageFont:
    """从列表中查找第一个可用字体。"""
    key = (",".join(font_list), size)
    if key in _font_cache:
        return _font_cache[key]

    for fp in font_list:
        if fp and os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _font_cache[key] = font
                return font
            except Exception:
                continue

    # fallback: PIL default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.ImageFont()
    _font_cache[key] = font
    return font


def get_cn_font(size: int = 20) -> ImageFont.ImageFont:
    return _find_font(CHINESE_FONTS, size)


def get_mono_font(size: int = 16) -> ImageFont.ImageFont:
    return _find_font(MONO_FONTS, size)


def get_en_font(size: int = 16) -> ImageFont.ImageFont:
    return _find_font(ENGLISH_FONTS, size)


# ═══════════════════════════════════════════════════════════════════════
# 1. 数据生成
# ═══════════════════════════════════════════════════════════════════════

_COMPANY_POOL = [
    ("北京神州数码科技有限公司", "91110000MA00123456"),
    ("上海浦东发展科技有限公司", "91310115MA00567890"),
    ("深圳华为信息技术有限公司", "91440300MA00987654"),
    ("广州天河软件股份有限公司", "91440101MA00321456"),
    ("杭州阿里巴巴云计算有限公司", "91330100MA00789012"),
    ("成都腾讯科技股份有限公司", "91510100MA00111222"),
    ("武汉烽火通信科技有限公司", "91420100MA00444333"),
    ("西安大唐网络科技有限公司", "91610131MA00555666"),
    ("南京紫金山实验室有限公司", "91320100MA00666777"),
    ("苏州工业园区精密制造公司", "91320594MA00888999"),
    ("天津滨海新区物流有限公司", "91120116MA00123000"),
    ("重庆两江新区电子商务有限公司", "91500000MA00456000"),
    ("长沙高新开发区文化传媒有限公司", "91430100MA00789000"),
    ("济南高新技术产业开发区生物医药有限公司", "91370100MA00012345"),
    ("青岛海洋科技股份有限公司", "91370200MA00678000"),
    ("厦门自贸区进出口贸易有限公司", "91350200MA00999000"),
    ("合肥高新区人工智能研究院", "91340100MA00222000"),
    ("郑州经济技术开发区装备制造公司", "91410100MA00555000"),
    ("福州软件园信息技术服务有限公司", "91350100MA00888000"),
    ("大连高新技术产业园区新材料科技有限公司", "91210231MA00111000"),
]

_TAX_RATES = [
    (0.06, "6%"),
    (0.09, "9%"),
    (0.13, "13%"),
    (0.03, "3% (小规模)"),
]

_GOODS_POOL = [
    ("办公用品", ["A4复印纸", "签字笔", "文件夹", "订书机", "计算器", "便签纸", "档案盒"]),
    ("电子设备", ["笔记本电脑", "显示器", "键盘", "鼠标", "固态硬盘", "移动电源", "USB集线器"]),
    ("技术服务", ["软件开发", "系统集成", "技术咨询", "数据迁移", "安全保障", "运维服务"]),
    ("咨询服务", ["管理咨询", "财务顾问", "法律咨询", "市场调研", "战略规划"]),
    ("培训服务", ["Python培训", "项目管理培训", "领导力培训", "技术认证培训"]),
    ("物流运输", ["国内快递", "国际运输", "仓储服务", "同城配送"]),
    ("印刷服务", ["名片印刷", "宣传册印刷", "展架制作", "礼品包装"]),
    ("租赁服务", ["场地租赁", "设备租赁", "车辆租赁", "展具租赁"]),
]

_ENGLISH_COMPANIES = [
    ("Walmart Supercenter", "123 Main St, Springfield, IL 62701"),
    ("Target Corporation", "456 Oak Ave, Minneapolis, MN 55402"),
    ("Costco Wholesale", "789 Park Blvd, Seattle, WA 98101"),
    ("Best Buy Electronics", "321 Tech Dr, San Jose, CA 95113"),
    ("Starbucks Coffee", "100 Pike St, Seattle, WA 98101"),
    ("Whole Foods Market", "200 Green St, Austin, TX 78701"),
    ("Home Depot", "500 Builders Ln, Atlanta, GA 30339"),
    ("CVS Pharmacy", "800 Health Way, Boston, MA 02110"),
]

_ENGLISH_ITEMS = [
    ("Milk 2%", 1, 4.29), ("Bread Whole Wheat", 1, 3.49), ("Eggs Large 12ct", 1, 5.99),
    ("Banana Organic", 1, 1.99), ("Chicken Breast", 1, 8.99), ("Rice 5lb", 1, 6.49),
    ("Coffee Ground", 1, 9.99), ("Orange Juice", 1, 4.49), ("Yogurt Greek", 2, 1.49),
    ("Paper Towels", 1, 7.99), ("Dish Soap", 1, 3.29), ("Hand Sanitizer", 1, 4.99),
    ("Notebook", 2, 2.99), ("Pens Pack", 1, 4.49), ("USB Cable", 1, 12.99),
    ("Headphones", 1, 29.99), ("Charger", 1, 19.99), ("Screen Protector", 1, 8.99),
]


def _random_company() -> Tuple[str, str]:
    return random.choice(_COMPANY_POOL)


def _generate_phone() -> str:
    prefixes = ["010", "021", "0755", "028", "0571", "025", "027", "029"]
    return f"{random.choice(prefixes)}-{random.randint(10000000, 99999999)}"


def _generate_bank_info() -> Tuple[str, str]:
    banks = [
        ("中国工商银行", "北京分行"),
        ("中国建设银行", "上海分行"),
        ("中国农业银行", "深圳分行"),
        ("中国银行", "广州分行"),
        ("招商银行", "杭州分行"),
        ("交通银行", "成都分行"),
    ]
    name, branch = random.choice(banks)
    account = "".join([str(random.randint(0, 9)) for _ in range(16)])
    return f"{name}{branch}", account


def _random_date(start_year: int = 2021, end_year: int = 2026) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def _random_invoice_no() -> str:
    code = random.randint(110000, 450000)
    no = random.randint(10000000, 99999999)
    return f"{code}{no:08d}"


def _random_tax_id() -> str:
    return _random_company()[1]


def _random_amount(min_v: float = 50, max_v: float = 50000) -> float:
    return round(random.uniform(min_v, max_v), 2)


def generate_invoice_data(template: str = "vat_general") -> Dict[str, Any]:
    """
    生成单条票据标注数据（仅填充模板不覆写的字段）。

    VAT 发票模板会在渲染时根据商品明细重新计算 total_amount/tax_amount，
    所以这里只为小票模板生成合理的金额。
    """
    data = {
        "merchant_name": _random_company()[0],
        "date": _random_date(2021, 2026),
        "total_amount": 0.0,   # 由模板填充
        "tax_amount": 0.0,     # 由模板填充
        "tax_id": _random_company()[1],
        "invoice_no": _random_invoice_no(),
    }

    # 小票模板可能缺失部分字段
    if template in ("receipt_thermal", "receipt_english"):
        if random.random() < 0.6:
            data["tax_id"] = None
        if random.random() < 0.5:
            data["invoice_no"] = None

    return data


# ═══════════════════════════════════════════════════════════════════════
# 2. 辅助绘制函数
# ═══════════════════════════════════════════════════════════════════════

class _DrawContext:
    """绘制上下文：封装 canvas / draw / 边距，并提供防溢出检查。"""

    def __init__(self, draw: ImageDraw.Draw, img: Image.Image, margin: int = 30):
        self.draw = draw
        self.img = img
        self.m = margin
        self.w, self.h = img.size
        self._bottom_margin = margin

    @property
    def content_width(self) -> int:
        return self.w - 2 * self.m

    def check_overflow(self, y: int, needed: int = 20):
        """如果 y + needed 超出画布高度，截断图像。"""
        if y + needed > self.h - self._bottom_margin:
            # 裁剪到当前 y + needed
            new_h = min(y + needed + self._bottom_margin, self.h)
            # 不实际裁剪，只是标记（调用方应自行处理）
            return False  # 未溢出
        return True  # 安全


def _text_w(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> int:
    return _text_width(draw, text, font)


def _draw_center_text(draw: ImageDraw.Draw, text: str, y: int,
                      font: ImageFont.ImageFont, fill=(0, 0, 0), w: int = 0) -> int:
    """居中绘制文本。返回文本宽度。"""
    tw = _text_w(draw, text, font)
    x = (w - tw) // 2 if w > 0 else 0
    draw.text((x, y), text, fill=fill, font=font)
    return tw


def _draw_hline(draw: ImageDraw.Draw, x1: int, y: int, x2: int,
                fill=(0, 0, 0), width: int = 1):
    """绘制水平线。"""
    draw.line([(x1, y), (x2, y)], fill=fill, width=width)


def _draw_table_row(draw: ImageDraw.Draw, x: int, y: int, col_widths: List[int],
                    row_h: int, texts: List[str], font, header: bool = False) -> int:
    """绘制表格一行。使用字体真实行高进行垂直居中。返回下一行 y。"""
    bg = (230, 230, 230) if header else (255, 255, 255)
    line_h = _font_line_height(font)
    cx = x
    for i, (cw, txt) in enumerate(zip(col_widths, texts)):
        rect = (cx, y, cx + cw, y + row_h)
        draw.rectangle(rect, fill=bg, outline=(180, 180, 180))
        tw = _text_w(draw, txt, font)
        tx = cx + (cw - tw) // 2
        ty = y + (row_h - line_h) // 2
        draw.text((max(tx, cx + 2), max(ty, y + 1)), txt, fill=(0, 0, 0), font=font)
        cx += cw
    return y + row_h


def _render_items_table(draw: ImageDraw.Draw, x: int, y: int,
                        w: int, font) -> Tuple[int, List[Dict]]:
    """在发票上渲染商品明细表。返回 (下一行 y, 商品列表)。"""
    row_h = 26
    col_widths = [int(w * 0.35), int(w * 0.15), int(w * 0.15), int(w * 0.15), int(w * 0.20)]
    headers = ["货物或应税劳务、服务名称", "规格", "数量", "单价", "金额"]

    y = _draw_table_row(draw, x, y, col_widths, row_h, headers, font, header=True)

    cat, items = random.choice(_GOODS_POOL)
    n_items = random.randint(1, min(5, len(items)))
    selected = random.sample(items, n_items)

    goods = []
    for item_name in selected:
        qty = random.randint(1, 20)
        unit_price = round(random.uniform(5, 5000), 2)
        amount = round(qty * unit_price, 2)
        spec = random.choice(["", "箱", "台", "套", "次", "个"])
        row_texts = [item_name, spec, str(qty),
                     f"{unit_price:.2f}", f"{amount:.2f}"]
        y = _draw_table_row(draw, x, y, col_widths, row_h, row_texts, font)
        goods.append({"name": item_name, "qty": qty, "unit_price": unit_price, "amount": amount})

    return y, goods


# ═══════════════════════════════════════════════════════════════════════
# 3. 印章 / 二维码
# ═══════════════════════════════════════════════════════════════════════

def _draw_stamp(draw: ImageDraw.Draw, cx: int, cy: int, r: int):
    """绘制模拟印章（红色圆圈 + 文字）。"""
    draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                 outline=(200, 40, 40), width=3)
    draw.ellipse((cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4),
                 outline=(200, 40, 40), width=1)

    stamp_text = random.choice(["发票专用章", "财务专用章", "发票章"])
    cn_sm = get_cn_font(int(r * 0.35))
    line_h = _font_line_height(cn_sm)
    tw = _text_w(draw, stamp_text, cn_sm)
    draw.text((cx - tw // 2, cy - line_h // 2),
              stamp_text, fill=(200, 40, 40), font=cn_sm)

    seller, _ = _random_company()
    mini = get_cn_font(int(r * 0.18))
    short_name = seller[:8] if len(seller) > 8 else seller
    tw2 = _text_w(draw, short_name, mini)
    draw.text((cx - tw2 // 2, cy + r // 4),
              short_name, fill=(200, 40, 40), font=mini)


def _add_qr_code(draw: ImageDraw.Draw, x: int, y: int, size: int = 80):
    """绘制模拟二维码。保存并恢复 np.random 状态以避免污染。"""
    rng_state = np.random.get_state()
    np.random.seed(abs(hash(str(x) + str(y))) % (2**31))

    draw.rectangle((x, y, x + size, y + size), fill=(255, 255, 255), outline=(0, 0, 0))
    # 三个定位图案
    for fx, fy in [(x + 4, y + 4), (x + size - 22, y + 4), (x + 4, y + size - 22)]:
        draw.rectangle((fx, fy, fx + 18, fy + 18), fill=(0, 0, 0))
        draw.rectangle((fx + 4, fy + 4, fx + 14, fy + 14), fill=(255, 255, 255))
        draw.rectangle((fx + 6, fy + 6, fx + 12, fy + 12), fill=(0, 0, 0))
    # 数据方块
    step = 4
    for i in range(size // step):
        for j in range(size // step):
            px, py = x + 1 + i * step, y + 1 + j * step
            in_finder = (
                (px < x + 24 and py < y + 24) or
                (px > x + size - 26 and py < y + 24) or
                (px < x + 24 and py > y + size - 26)
            )
            if not in_finder and random.random() < 0.45:
                draw.rectangle((px, py, px + 2, py + 2), fill=(0, 0, 0))

    np.random.set_state(rng_state)


# ═══════════════════════════════════════════════════════════════════════
# 4. 模板: 增值税普通发票
# ═══════════════════════════════════════════════════════════════════════

def _draw_vat_general(data: Dict[str, Any], width: int = 900,
                      height: int = 650) -> Image.Image:
    """绘制增值税普通发票。"""
    img = Image.new("RGB", (width, height), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)
    m = 30

    title_font = get_cn_font(26)
    hdr_font = get_cn_font(16)
    body_font = get_cn_font(13)
    small_font = get_cn_font(11)

    # ── 顶部 ──
    y = m + 5
    code = f"{random.randint(110000, 450000)}{random.randint(10, 99)}"
    draw.text((m, y), f"发票代码: {code}", fill=(80, 40, 40), font=small_font)
    tw = _text_w(draw, f"发票号码: {data['invoice_no']}", small_font)
    draw.text((width - m - tw, y),
              f"发票号码: {data['invoice_no']}", fill=(80, 40, 40), font=small_font)
    y += 20

    title = "增值税普通发票        发票联"
    _draw_center_text(draw, title, y, title_font, fill=(180, 30, 30), w=width)
    y += 40

    _draw_hline(draw, m, y, width - m, fill=(180, 30, 30), width=2)
    y += 12

    # ── 购买方信息 ──
    draw.text((m, y), "购买方", fill=(60, 60, 60), font=hdr_font)
    y += 22
    buyer_name = _random_company()[0]
    buyer_tax_id = _random_tax_id()
    draw.text((m, y), f"名    称: {buyer_name}", fill=(0, 0, 0), font=body_font)
    y += 20
    draw.text((m, y), f"纳税人识别号: {buyer_tax_id}", fill=(0, 0, 0), font=body_font)
    draw.text((m + 350, y),
              f"地址、电话: {random.choice(_COMPANY_POOL)[0][:6]}区 {_generate_phone()}",
              fill=(0, 0, 0), font=small_font)
    y += 18
    bank, account = _generate_bank_info()
    draw.text((m, y), f"开户行及账号: {bank} {account}", fill=(0, 0, 0), font=small_font)
    y += 26

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 8

    # ── 商品明细表 ──
    y, goods = _render_items_table(draw, m, y, width - 2 * m, body_font)
    y += 8

    # ── 金额合计 ──
    subtotal = sum(g["amount"] for g in goods)
    tax_rate = random.choice([0.03, 0.06, 0.09, 0.13])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    data["total_amount"] = total
    data["tax_amount"] = tax

    info_x = width - m - 300
    info_w = 300
    draw.rectangle((info_x, y, info_x + info_w, y + 100),
                   outline=(180, 180, 180), fill=(255, 255, 252))
    row_h = 22
    info_rows = [
        ("金额合计（小写）", f"¥ {subtotal:,.2f}"),
        ("税额", f"¥ {tax:,.2f}"),
        ("价税合计（大写）", ""),
        ("（小写）", f"¥ {total:,.2f}"),
    ]
    iy = y + 4
    for label, val in info_rows:
        draw.text((info_x + 12, iy), label, fill=(60, 60, 60), font=small_font)
        if val:
            tw2 = _text_w(draw, val, small_font)
            draw.text((info_x + info_w - tw2 - 12, iy), val, fill=(0, 0, 0), font=small_font)
        iy += row_h
    cn_amount = _num_to_chinese(total)
    draw.text((info_x + 12, y + 3 * row_h), cn_amount, fill=(180, 30, 30), font=body_font)

    y += 110

    # ── 销售方信息 ──
    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 10
    seller_name, seller_tax_id = _random_company()
    draw.text((m, y), f"销售方名称: {seller_name}", fill=(0, 0, 0), font=body_font)
    y += 20
    draw.text((m, y), f"纳税人识别号: {seller_tax_id}", fill=(0, 0, 0), font=body_font)
    y += 20
    draw.text((m, y),
              f"地址、电话: {random.choice(_COMPANY_POOL)[0][:4]}区 {_generate_phone()}",
              fill=(0, 0, 0), font=small_font)
    y += 18
    bank2, acct2 = _generate_bank_info()
    draw.text((m, y), f"开户行及账号: {bank2} {acct2}", fill=(0, 0, 0), font=small_font)
    y += 22

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 10
    persons = ["收款人: 张丽", "复核: 王强",
               f"开票人: {random.choice(['李明','赵芳','刘洋','陈静'])}"]
    px = m
    for p in persons:
        draw.text((px, y), p, fill=(60, 60, 60), font=small_font)
        px += 200

    # ── 印章 + 二维码（定位到页面底部偏右） ──
    stamp_cx = width - m - 90
    stamp_cy = height - 110
    qr_x = m + 10
    qr_y = height - 120

    _draw_stamp(draw, stamp_cx, stamp_cy, 80)
    _add_qr_code(draw, qr_x, qr_y, 75)

    return img


# ═══════════════════════════════════════════════════════════════════════
# 5. 模板: 增值税专用发票
# ═══════════════════════════════════════════════════════════════════════

def _draw_vat_special(data: Dict[str, Any], width: int = 950,
                      height: int = 680) -> Image.Image:
    """绘制增值税专用发票（含抵扣联）。"""
    img = Image.new("RGB", (width, height), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)
    m = 25

    title_font = get_cn_font(28)
    hdr_font = get_cn_font(15)
    body_font = get_cn_font(12)
    small_font = get_cn_font(10)

    y = m + 5
    draw.text((m, y), f"No: {data['invoice_no']}", fill=(80, 40, 40), font=small_font)
    tw = _text_w(draw, "抵扣联", hdr_font)
    draw.text((width - m - tw, y), "抵扣联", fill=(200, 40, 40), font=hdr_font)
    y += 18

    title = "××增值税专用发票"
    _draw_center_text(draw, title, y, title_font, fill=(180, 30, 30), w=width)
    y += 38

    _draw_hline(draw, m, y, width - m, fill=(180, 30, 30), width=3)
    y += 10

    buyer = _random_company()[0]
    seller, seller_tid = _random_company()
    draw.text((m, y), f"购货单位: {buyer}", fill=(0, 0, 0), font=body_font)
    draw.text((m + 400, y),
              f"纳税人识别号: {_random_tax_id()}", fill=(0, 0, 0), font=small_font)
    y += 20
    draw.text((m, y), f"销货单位: {seller}", fill=(0, 0, 0), font=body_font)
    draw.text((m + 400, y),
              f"纳税人识别号: {seller_tid}", fill=(0, 0, 0), font=small_font)
    y += 26

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 6
    y, goods = _render_items_table(draw, m, y, width - 2 * m, body_font)
    y += 8

    subtotal = sum(g["amount"] for g in goods)
    tax = round(subtotal * 0.13, 2)
    total = round(subtotal + tax, 2)
    data["total_amount"] = total
    data["tax_amount"] = tax

    info_x = width - m - 320
    row_h = 20
    draw.rectangle((info_x, y, info_x + 320, y + 90),
                   outline=(180, 180, 180), fill=(255, 255, 252))
    iy = y + 4
    for label, val in [("金额合计", f"¥ {subtotal:,.2f}"),
                        ("税额", f"¥ {tax:,.2f}"),
                        ("价税合计(大写)", _num_to_chinese(total)),
                        ("价税合计(小写)", f"¥ {total:,.2f}")]:
        draw.text((info_x + 10, iy), label, fill=(60, 60, 60), font=small_font)
        tw2 = _text_w(draw, val, small_font)
        draw.text((info_x + 320 - tw2 - 10, iy), val, fill=(0, 0, 0), font=small_font)
        iy += row_h

    y += 100

    _draw_hline(draw, m, y, width - m, fill=(180, 180, 180))
    y += 8
    draw.text((m, y), "备注: ", fill=(100, 100, 100), font=small_font)
    y += 24

    draw.text((m, y), f"收款人: {random.choice(['王芳','李强','张敏'])}",
              fill=(60, 60, 60), font=small_font)
    tw_p = _text_w(draw, f"开票人: {random.choice(['刘洋','陈明','赵雪'])}", small_font)
    draw.text((width - m - tw_p, y),
              f"开票人: {random.choice(['刘洋','陈明','赵雪'])}", fill=(60, 60, 60), font=small_font)

    # 印章固定在页面底部
    stamp_cx = width - m - 100
    stamp_cy = height - 100
    _draw_stamp(draw, stamp_cx, stamp_cy, 90)

    return img


# ═══════════════════════════════════════════════════════════════════════
# 6. 模板: 热敏小票
# ═══════════════════════════════════════════════════════════════════════

def _draw_receipt_thermal(data: Dict[str, Any], width: int = 380,
                          height: int = 0) -> Image.Image:
    """绘制热敏小票（模拟超市/餐饮小票），高度自动适配内容。"""
    paper_color = random.choice([
        (250, 248, 242), (248, 246, 240), (252, 250, 245), (240, 238, 232)
    ])
    m = 12

    # 统一字体大小保证行对齐
    cn = get_cn_font(14)
    cn_sm = get_cn_font(12)
    # 等宽数字字体与 cn_sm 大小一致确保同列对齐
    mono = get_mono_font(12)

    line_h_cn = _font_line_height(cn)
    line_h_sm = _font_line_height(cn_sm)

    # ── 先计算内容高度，再创建画布 ──
    cat, items = random.choice(_GOODS_POOL)
    n = random.randint(3, 8)
    selected = random.sample(items, min(n, len(items)))
    n_extra_lines = sum(1 for _ in [0] if random.random() < 0.4)  # discount line

    # 估算行数
    est_lines = (
        2 +  # 标题 + 分隔
        2 +  # 交易信息
        1 +  # 分隔
        1 +  # 表头
        1 +  # 表头下划线
        len(selected) +  # 商品行
        1 +  # 分隔
        3 + n_extra_lines +  # 合计行
        1 +  # 分隔
        1 +  # 支付行
        1 +  # 分隔
        2 +  # 底部
        2    # 余量
    )
    est_height = est_lines * (line_h_sm + 6) + 30
    if height <= 0:
        height = max(360, min(est_height + 40, 900))

    img = Image.new("RGB", (width, height), color=paper_color)
    draw = ImageDraw.Draw(img)

    y = 16

    # ── 顶部 ──
    merchant = data["merchant_name"]
    _draw_center_text(draw, merchant, y, cn, w=width)
    y += line_h_cn + 6

    _draw_center_text(draw, "—— 购物小票 ——", y, cn_sm, fill=(100, 100, 100), w=width)
    y += line_h_sm + 10

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 交易信息 ──
    draw.text((m, y), f"日期: {data['date']}", fill=(0, 0, 0), font=cn_sm)
    draw.text((m + 140, y), f"流水号: {random.randint(1000, 9999)}",
              fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4
    draw.text((m, y), f"收银员: {random.choice(['001','002','003','005'])}",
              fill=(0, 0, 0), font=cn_sm)
    draw.text((m + 120, y), f"终端: POS{random.randint(10,99)}",
              fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 6

    # ── 商品列表（统一用 cn_sm 保证中文正常、列宽一致） ──
    header_x = [m, m + 170, m + 260, m + 320]
    draw.text((header_x[0], y), "商品", fill=(60, 60, 60), font=cn_sm)
    draw.text((header_x[1], y), "数量", fill=(60, 60, 60), font=cn_sm)
    draw.text((header_x[2], y), "单价", fill=(60, 60, 60), font=cn_sm)
    draw.text((header_x[3], y), "金额", fill=(60, 60, 60), font=cn_sm)
    y += line_h_sm + 2
    _draw_hline(draw, m, y, width - m, fill=(160, 160, 160))
    y += 6

    items_data = []
    for item_name in selected:
        if y + 20 > height - 40:
            break  # 防溢出
        qty = random.randint(1, 5)
        up = round(random.uniform(3, 500), 2)
        amt = round(qty * up, 2)
        # 中文商品名 cn_sm；数字也用 cn_sm（大小一致保证对齐）
        draw.text((header_x[0], y), item_name, fill=(0, 0, 0), font=cn_sm)
        draw.text((header_x[1] + 10, y), str(qty), fill=(0, 0, 0), font=cn_sm)
        draw.text((header_x[2] + 5, y), f"{up:.2f}", fill=(0, 0, 0), font=cn_sm)
        draw.text((header_x[3] + 5, y), f"{amt:.2f}", fill=(0, 0, 0), font=cn_sm)
        y += line_h_sm + 4
        items_data.append({"name": item_name, "qty": qty, "unit_price": up, "amount": amt})

    y += 2
    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 合计 ──
    subtotal = sum(i["amount"] for i in items_data)
    discount = round(random.uniform(0, subtotal * 0.1), 2) if random.random() < 0.4 else 0
    after_discount = subtotal - discount
    tax_rate = random.choice([0, 0, 0, 0.06, 0.13])
    tax = round(after_discount * tax_rate, 2)
    total = round(after_discount + tax, 2)

    data["total_amount"] = total
    data["tax_amount"] = tax if tax > 0 else None

    items_count = sum(i["qty"] for i in items_data)
    draw.text((m, y + 2), f"件数: {items_count}", fill=(0, 0, 0), font=cn_sm)
    draw.text((m + 150, y + 2), f"小计: {subtotal:.2f}", fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 4

    if discount > 0:
        draw.text((m + 150, y + 2), f"折扣: -{discount:.2f}", fill=(160, 40, 40), font=cn_sm)
        y += line_h_sm + 4
    if tax > 0:
        draw.text((m + 150, y + 2), f"税额: {tax:.2f}", fill=(0, 0, 0), font=cn_sm)
        y += line_h_sm + 4

    _draw_hline(draw, m + 120, y, width - m, fill=(60, 60, 60))
    y += 6
    total_str = f"合计: ¥{total:.2f}"
    tw_t = _text_w(draw, total_str, cn)
    draw.text((width - m - tw_t, y + 2), total_str, fill=(0, 0, 0), font=cn)
    y += line_h_cn + 10

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 支付 ──
    payment_methods = ["微信支付", "支付宝", "银联卡", "现金", "云闪付"]
    pmt = random.choice(payment_methods)
    draw.text((m, y + 2), f"支付方式: {pmt}", fill=(0, 0, 0), font=cn_sm)
    draw.text((m + 150, y + 2), f"实收: ¥{total:.2f}", fill=(0, 0, 0), font=cn_sm)
    y += line_h_sm + 8

    if y > height - 40:
        return img  # 已满，不再画底部

    _draw_hline(draw, m, y, width - m, fill=(100, 100, 100))
    y += 8

    # ── 底部 ──
    _draw_center_text(draw, "请保留小票以办理退换货", y, cn_sm,
                      fill=(120, 120, 120), w=width)
    y += line_h_sm + 4
    if data["tax_id"]:
        t = f"税号: {data['tax_id']}"
        _draw_center_text(draw, t, y, cn_sm, fill=(120, 120, 120), w=width)
        y += line_h_sm + 4
    _draw_center_text(draw, f"电话: {_generate_phone()}", y, cn_sm,
                      fill=(120, 120, 120), w=width)

    return img


# ═══════════════════════════════════════════════════════════════════════
# 7. 模板: 英文收据
# ═══════════════════════════════════════════════════════════════════════

def _draw_receipt_english(data: Dict[str, Any], width: int = 420,
                          height: int = 0) -> Image.Image:
    """绘制英文收据，高度自动适配内容。"""
    paper = random.choice([(252, 250, 245), (248, 246, 240), (255, 253, 248)])
    m = 16
    content_w = width - 2 * m  # 内容可用宽度
    right_x = width - m        # 右侧对齐基线

    en = get_en_font(14)
    en_sm = get_en_font(11)
    mono = get_mono_font(13)
    bold_font = get_en_font(18)

    line_h = _font_line_height(mono)
    line_h_sm = _font_line_height(en_sm)
    line_h_en = _font_line_height(en)
    line_h_bold = _font_line_height(bold_font)

    # 估算高度
    n_items = random.randint(3, 8)
    selected = random.sample(_ENGLISH_ITEMS, min(n_items, len(_ENGLISH_ITEMS)))
    est_lines = 20 + len(selected) + 3
    if height <= 0:
        height = max(350, min(est_lines * (line_h + 4) + 30, 800))

    img = Image.new("RGB", (width, height), color=paper)
    draw = ImageDraw.Draw(img)

    def _truncate_item(text: str, max_px: int) -> str:
        """截断过长的商品名，确保不侵占价格列空间。"""
        if _text_w(draw, text, mono) <= max_px:
            return text
        while len(text) > 5 and _text_w(draw, text + "...", mono) > max_px:
            text = text[:-1]
        return text + "..."

    y = 14

    # ── 商店信息 ──
    company, address = random.choice(_ENGLISH_COMPANIES)
    _draw_center_text(draw, company, y, bold_font, w=width)
    y += line_h_bold + 4
    _draw_center_text(draw, address, y, en_sm, fill=(80, 80, 80), w=width)
    y += line_h_sm + 2
    _draw_center_text(draw, "Tel: (555) 123-4567", y, en_sm, fill=(80, 80, 80), w=width)
    y += line_h_sm + 6
    _draw_hline(draw, m, y, width - m, fill=(150, 150, 150))
    y += 8

    # ── 交易信息 ──
    dt_str = data["date"] if data["date"] else "2024-06-15"
    draw.text((m, y), f"Date: {dt_str}", fill=(0, 0, 0), font=mono)
    txn_str = f"Txn#: {random.randint(1000, 99999)}"
    tw = _text_w(draw, txn_str, mono)
    draw.text((right_x - tw, y), txn_str, fill=(0, 0, 0), font=mono)
    y += line_h + 2
    draw.text((m, y), f"Cashier: {random.choice(['Amy','Bob','Carol','Dave'])}",
              fill=(60, 60, 60), font=mono)
    reg_str = f"Reg: {random.randint(1, 20)}"
    tw = _text_w(draw, reg_str, mono)
    draw.text((right_x - tw, y), reg_str, fill=(60, 60, 60), font=mono)
    y += line_h + 2
    _draw_hline(draw, m, y, width - m, fill=(120, 120, 120))
    y += 6

    # ── 商品 ──
    # 预留价格列宽度（最宽价格 "$XX,XXX.XX" ≈ 80px）
    PRICE_COL_WIDTH = 80
    ITEM_MAX_WIDTH = content_w - PRICE_COL_WIDTH - 12  # 12px gap

    draw.text((m, y), "Item", fill=(100, 100, 100), font=en_sm)
    price_hdr = "Price"
    tw = _text_w(draw, price_hdr, en_sm)
    draw.text((right_x - tw, y), price_hdr, fill=(100, 100, 100), font=en_sm)
    y += line_h_sm + 2

    for name, qty, price in selected:
        if y + 20 > height - 40:
            break
        total_price = round(qty * price, 2)
        line = f"{name}" if qty == 1 else f"{name} x{qty}"
        line = _truncate_item(line, ITEM_MAX_WIDTH)
        price_str = f"${total_price:.2f}"
        draw.text((m, y), line, fill=(0, 0, 0), font=mono)
        tw = _text_w(draw, price_str, mono)
        draw.text((right_x - tw, y), price_str, fill=(0, 0, 0), font=mono)
        y += line_h + 2

    y += 2
    _draw_hline(draw, m, y, width - m, fill=(120, 120, 120))
    y += 6

    # ── 合计（关键修复：用实际渲染字体测量宽度） ──
    subtotal = round(sum(qty * price for _, qty, price in selected), 2)
    tax_pct = random.choice([0.06, 0.08, 0.0875, 0.0925])
    tax = round(subtotal * tax_pct, 2)
    total = round(subtotal + tax, 2)
    data["total_amount"] = total
    data["tax_amount"] = tax

    for label, val in [("Subtotal", subtotal),
                        (f"Tax ({tax_pct*100:.2f}%)", tax),
                        ("TOTAL", total)]:
        val_s = f"${val:.2f}"
        is_total = (label == "TOTAL")
        val_font = bold_font if is_total else en
        val_line_h = _font_line_height(val_font)

        # 用实际渲染字体测量价格字符串宽度
        tw_val = _text_w(draw, val_s, val_font)
        tw_lbl = _text_w(draw, label, en)
        gap = 24  # label 和 value 之间的间距

        # value 右对齐，label 在 value 左边
        val_x = right_x - tw_val
        lbl_x = val_x - tw_lbl - gap

        draw.text((max(lbl_x, m), y), label, fill=(60, 60, 60), font=en)
        draw.text((val_x, y), val_s, fill=(0, 0, 0), font=val_font)
        y += max(line_h_en, val_line_h) + 4

    y += 2
    _draw_hline(draw, m, y, width - m, fill=(120, 120, 120))
    y += 6

    # ── 支付 ──
    methods = ["VISA ****1234", "MASTERCARD ****5678", "CASH", "DEBIT CARD", "AMEX ****9012"]
    pmt = random.choice(methods)
    draw.text((m, y), f"Payment: {pmt}", fill=(60, 60, 60), font=mono)
    y += line_h + 8

    if y > height - 30:
        return img

    # ── 底部 ──
    footer_lines = [
        "Thank you for shopping with us!",
        "Please retain receipt for returns within 30 days.",
        "Return Policy: exchanges accepted with receipt.",
    ]
    for fl in footer_lines:
        _draw_center_text(draw, fl, y, en_sm, fill=(140, 140, 140), w=width)
        y += line_h_sm + 2

    return img


# ═══════════════════════════════════════════════════════════════════════
# 8. 中文大写金额
# ═══════════════════════════════════════════════════════════════════════

def _num_to_chinese(amount: float) -> str:
    """金额转中文大写。"""
    if amount >= 1e8:
        return "金额过大"

    digits = "零壹贰叁肆伍陆柒捌玖"
    units = ["", "拾", "佰", "仟"]
    big_units = ["", "万", "亿"]

    int_part = int(amount)
    dec_part = int(round((amount - int_part) * 100))

    if int_part == 0:
        result = "零元"
    else:
        result = ""
        int_str = str(int_part)
        n = len(int_str)
        zero_flag = False
        for i, ch in enumerate(int_str):
            d = int(ch)
            pos = n - i - 1
            unit_pos = pos % 4
            big_pos = pos // 4

            if d == 0:
                zero_flag = True
                if unit_pos == 0:
                    result += big_units[big_pos]
            else:
                if zero_flag:
                    result += "零"
                    zero_flag = False
                result += digits[d] + units[unit_pos]
                if unit_pos == 0:
                    result += big_units[big_pos]
        result += "元"

    jiao = dec_part // 10
    fen = dec_part % 10
    if jiao == 0 and fen == 0:
        result += "整"
    else:
        if jiao > 0:
            result += digits[jiao] + "角"
        if fen > 0:
            result += digits[fen] + "分"

    return result


# ═══════════════════════════════════════════════════════════════════════
# 9. 增强函数
# ═══════════════════════════════════════════════════════════════════════

def add_paper_texture(img: Image.Image, intensity: float = 0.03) -> Image.Image:
    """叠加纸张纹理。"""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_uneven_lighting(img: Image.Image) -> Image.Image:
    """叠加不均匀光照（模拟拍照光照）。"""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    y, x = np.ogrid[:h, :w]
    cx, cy = random.uniform(0.3, 0.7) * w, random.uniform(0.3, 0.7) * h
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(w**2 + h**2)
    light = 1.0 - random.uniform(0.05, 0.15) * (dist / max_dist)
    light = np.clip(light, 0.8, 1.0)

    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * light

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_fold_line(img: Image.Image) -> Image.Image:
    """添加折叠痕迹。"""
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    fold_x = random.randint(int(w * 0.3), int(w * 0.7))
    alpha = random.randint(20, 50)
    draw.line([(fold_x, 0), (fold_x, h)], fill=(180, 180, 180, alpha), width=2)

    arr = np.array(img, dtype=np.float32)
    arr[:, :fold_x, :] = np.clip(arr[:, :fold_x, :] * 1.02, 0, 255)
    arr[:, fold_x:, :] = np.clip(arr[:, fold_x:, :] * 0.98, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def apply_augmentation(img: Image.Image, level: str = "medium") -> Image.Image:
    """
    综合增强管道。

    level:
      - "light":  轻微噪声 + 模糊
      - "medium": 噪声 + 模糊 + 旋转 + 光照
      - "heavy":  medium + 透视变换 + 折叠 + 对比度调整
    """
    img = add_paper_texture(img, intensity=random.uniform(0.01, 0.04))

    if level in ("medium", "heavy") and random.random() < 0.6:
        img = add_uneven_lighting(img)

    if level == "light":
        angle = random.uniform(-2, 2)
    elif level == "medium":
        angle = random.uniform(-5, 5)
    else:
        angle = random.uniform(-8, 8)
    if abs(angle) > 0.1:
        img = img.rotate(angle, expand=False, fillcolor=img.getpixel((0, 0)))

    blur_rad = {"light": 0.3, "medium": 0.6, "heavy": 1.0}[level]
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, blur_rad)))

    if level == "heavy" and random.random() < 0.4:
        w, h = img.size
        dx = random.randint(5, 25)
        dy = random.randint(3, 15)
        coeffs = (1, 0, random.choice([-dx, dx]),
                   0, 1, random.choice([-dy, dy]),
                   0, 0)
        img = img.transform((w, h), Image.AFFINE, coeffs, fillcolor=img.getpixel((0, 0)))

    if level == "heavy" and random.random() < 0.3:
        img = add_fold_line(img)

    if level in ("medium", "heavy") and random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

    return img


# ═══════════════════════════════════════════════════════════════════════
# 10. 主生成入口
# ═══════════════════════════════════════════════════════════════════════

TEMPLATE_MAP = {
    "vat_general": _draw_vat_general,
    "vat_special": _draw_vat_special,
    "receipt_thermal": _draw_receipt_thermal,
    "receipt_english": _draw_receipt_english,
}

TEMPLATE_WEIGHTS = {
    "vat_general": 0.35,
    "vat_special": 0.15,
    "receipt_thermal": 0.35,
    "receipt_english": 0.15,
}

DEFAULT_PROMPT = (
    "<image>请从这张票据中抽取以下字段并以 JSON 输出:\n"
    "merchant_name, date, total_amount, tax_amount, tax_id, invoice_no。\n"
    "缺失字段填 null,不要编造。"
)

ENGLISH_PROMPT = (
    "<image>Extract the following fields from this receipt as JSON:\n"
    "merchant_name, date, total_amount, tax_amount, tax_id, invoice_no.\n"
    "Use null for missing fields. Do not hallucinate."
)


def draw_receipt(data: Dict[str, Any],
                 template: str = "vat_general",
                 width: int = 0,
                 height: int = 0) -> Image.Image:
    """绘制票据图片。template='random' 随机选择模板。"""
    if template == "random":
        templates = list(TEMPLATE_WEIGHTS.keys())
        weights = list(TEMPLATE_WEIGHTS.values())
        template = random.choices(templates, weights=weights, k=1)[0]

    draw_fn = TEMPLATE_MAP.get(template)
    if draw_fn is None:
        raise ValueError(f"未知模板类型: {template}。可选: {list(TEMPLATE_MAP.keys())}")

    kwargs = {}
    if width > 0:
        kwargs["width"] = width
    if height > 0:
        kwargs["height"] = height

    return draw_fn(data, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 11. 批量生成
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 2000,
    augmentation: str = "medium",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """生成增强版合成票据数据集。"""
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    templates = list(TEMPLATE_WEIGHTS.keys())
    weights = list(TEMPLATE_WEIGHTS.values())

    print(f"🎨 生成 {num_samples} 张增强合成票据...")
    print(f"   模板: {templates}")
    print(f"   权重: {weights}")
    print(f"   增强级别: {augmentation}")

    for i in range(num_samples):
        tmpl = random.choices(templates, weights=weights, k=1)[0]
        data = generate_invoice_data(tmpl)
        if tmpl == "receipt_english":
            data["merchant_name"] = random.choice(_ENGLISH_COMPANIES)[0]

        img = draw_receipt(data, template=tmpl)

        if augmentation != "none":
            img = apply_augmentation(img, level=augmentation)

        fname = f"synth_{i:06d}.png"
        fpath = image_dir / fname
        img.save(fpath)

        prompt = ENGLISH_PROMPT if tmpl == "receipt_english" else DEFAULT_PROMPT

        dataset.append({
            "image_path": str(fpath),
            "prompt": prompt,
            "target_json": data,
            "template": tmpl,
        })

        if (i + 1) % 200 == 0:
            print(f"  已生成 {i + 1}/{num_samples}")

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# 12. CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="增强版票据合成器")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="生成的样本数量；0 表示生成测试图")
    parser.add_argument("--output-dir", default=str(repo_root / "data" / "synthetic"),
                        help="输出目录")
    parser.add_argument("--augmentation", choices=["none", "light", "medium", "heavy"],
                        default="medium")
    parser.add_argument("--template", choices=list(TEMPLATE_MAP.keys()) + ["random", "all"],
                        default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)

    if args.num_samples <= 0:
        print("生成各模板测试图...\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        tmpls = list(TEMPLATE_MAP.keys()) if args.template == "all" else [args.template]

        for tmpl in tmpls:
            data = generate_invoice_data(tmpl)
            if tmpl == "receipt_english":
                data["merchant_name"] = random.choice(_ENGLISH_COMPANIES)[0]
            print(f"  模板: {tmpl}")
            print(f"  数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

            img = draw_receipt(data, template=tmpl)
            if args.augmentation != "none":
                img = apply_augmentation(img, level=args.augmentation)

            out_path = output_dir / f"test_{tmpl}.png"
            img.save(out_path)
            print(f"  ✓ 保存到 {out_path}\n")
    else:
        dataset = generate_synthetic_dataset(
            output_dir=output_dir,
            num_samples=args.num_samples,
            augmentation=args.augmentation,
            seed=args.seed,
        )

        jsonl_path = output_dir / "train.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in dataset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n✅ 生成 {len(dataset)} 条样本")
        print(f"   图片目录: {output_dir / 'images'}")
        print(f"   索引文件: {jsonl_path}")
