"""Собрать итоговый отчёт по проекту LawanKZ в формате DOCX."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
TMP_DIR = ROOT / "tmp" / "report_artifacts"
REPORT_PATH = OUTPUT_DIR / "LawanKZ_итоговый_отчёт.docx"
RESULTS_1_4 = ROOT / "results" / "weeks_1_4"
RESULTS_5_6 = ROOT / "results" / "weeks_5_6"

NAVY = "1F4E78"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "D9D9D9"
CODE_GRAY = "F3F4F6"
BLACK = RGBColor(0, 0, 0)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=110, bottom=100, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), "6")
        tag.set(qn("w:color"), LIGHT_GRAY)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    header._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for index, text in enumerate(headers):
        cell = header.cells[index]
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9.5)
    for row_index, values in enumerate(rows):
        row = table.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.width = Inches(widths[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index % 2:
                set_cell_shading(cell, PALE_BLUE)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index == 0 else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(str(value))
            run.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_field(paragraph, instruction: str, display_text: str = "") -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = display_text
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def remove_paragraph_borders(paragraph_or_style) -> None:
    """Убрать у абзаца или стиля декоративные линии Word."""
    element = paragraph_or_style._element
    p_pr = element.get_or_add_pPr()
    border = p_pr.find(qn("w:pBdr"))
    if border is not None:
        p_pr.remove(border)


def add_hyperlink(paragraph, text: str, url: str) -> None:
    relationship_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.extend([props, text_node])
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_body(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Body Text"]
    if bold_lead and text.startswith(bold_lead):
        paragraph.add_run(bold_lead).bold = True
        paragraph.add_run(text[len(bold_lead):])
    else:
        paragraph.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Cm(0.75)
        paragraph.paragraph_format.first_line_indent = Cm(-0.35)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.2
        paragraph.add_run(item)


def add_code(doc: Document, code: str, caption: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Code Block"]
    run = paragraph.add_run(code.strip())
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    caption_paragraph = doc.add_paragraph(caption, style="Caption")
    caption_paragraph.paragraph_format.keep_with_next = False


def add_figure(doc: Document, image_path: Path, width_inches: float, caption: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    caption_paragraph = doc.add_paragraph(caption, style="Caption")
    caption_paragraph.paragraph_format.keep_with_next = False


def chapter(doc: Document, title: str, *, page_break: bool = False) -> None:
    if page_break:
        doc.add_page_break()
    doc.add_heading(title, level=1)


def make_pipeline_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    labels = [
        "Страницы НПА\nӘділет",
        "Извлечение\nстатей",
        "Очистка и\nлемматизация",
        "BoW TF IDF\nэмбеддинги",
        "Поиск похожих\nстатей",
    ]
    x = [0.08, 0.29, 0.5, 0.71, 0.92]
    for index, (position, label) in enumerate(zip(x, labels)):
        ax.text(
            position,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            bbox=dict(boxstyle="round,pad=0.65", facecolor="#1F4E78", edgecolor="#1F4E78"),
            transform=ax.transAxes,
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[index + 1] - 0.085, 0.5),
                xytext=(position + 0.085, 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.7, color="#5B6573"),
                xycoords=ax.transAxes,
            )
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(1.8)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12.5)
    normal.font.color.rgb = BLACK

    body = doc.styles["Body Text"]
    body.font.name = "Times New Roman"
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    body.font.size = Pt(12.5)
    body.font.color.rgb = BLACK
    body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body.paragraph_format.first_line_indent = Cm(1.25)
    body.paragraph_format.line_spacing = 1.15
    body.paragraph_format.space_after = Pt(4)
    body.paragraph_format.widow_control = True

    title = doc.styles["Title"]
    title.font.name = "Times New Roman"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = BLACK
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)
    remove_paragraph_borders(title)

    for style_name, size in (("Heading 1", 15), ("Heading 2", 13.5), ("Heading 3", 13)):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.keep_with_next = True

    for style_name, left_indent, size in (("TOC 1", 0, 10), ("TOC 2", 0.35, 9.5), ("TOC 3", 0.7, 9)):
        try:
            style = doc.styles[style_name]
        except KeyError:
            style = doc.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.color.rgb = BLACK
        style.paragraph_format.left_indent = Cm(left_indent)
        style.paragraph_format.space_after = Pt(1)
        style.paragraph_format.line_spacing = 1.0

    caption = doc.styles["Caption"]
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    caption.font.size = Pt(10)
    caption.font.italic = False
    caption.font.color.rgb = BLACK
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)

    code = doc.styles.add_style("Code Block", WD_STYLE_TYPE.PARAGRAPH)
    code.font.name = "Consolas"
    code._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    code.font.size = Pt(8.5)
    code.font.color.rgb = BLACK
    code.paragraph_format.left_indent = Cm(0.4)
    code.paragraph_format.right_indent = Cm(0.4)
    code.paragraph_format.space_before = Pt(5)
    code.paragraph_format.space_after = Pt(3)
    code.paragraph_format.line_spacing = 1.0
    p_pr = code._element.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), CODE_GRAY)
    p_pr.append(shading)

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = hp.add_run("Анализ текстов нормативных правовых актов")
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.font.color.rgb = BLACK

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(fp, "PAGE", "1")
    for run in fp.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)

    settings = doc.settings._element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)


def build_report() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = TMP_DIR / "pipeline.png"
    make_pipeline_figure(pipeline_path)

    corpus = json.loads((ROOT / "data" / "processed" / "corpus_summary.json").read_text(encoding="utf-8"))
    summary_1_4 = json.loads((RESULTS_1_4 / "analysis_summary.json").read_text(encoding="utf-8"))
    summary_5_6 = json.loads((RESULTS_5_6 / "analysis_summary.json").read_text(encoding="utf-8"))
    samples = json.loads((RESULTS_5_6 / "sample_search_results.json").read_text(encoding="utf-8"))

    doc = Document()
    configure_document(doc)
    props = doc.core_properties
    props.title = "Анализ текстов законов и нормативных правовых актов"
    props.subject = "Итоговый отчёт по учебному NLP проекту LawanKZ"
    props.author = "LawanKZ"
    props.keywords = "NLP, НПА, spaCy, TF-IDF, Word2Vec, GloVe, fastText"

    # Титульный лист
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(72)
    r = p.add_run("УЧЕБНЫЙ ОТЧЁТ ПО ОБРАБОТКЕ ЕСТЕСТВЕННОГО ЯЗЫКА")
    r.bold = True
    r.font.size = Pt(14)
    r.font.name = "Times New Roman"
    title = doc.add_paragraph("Анализ текстов законов и нормативных правовых актов", style="Title")
    remove_paragraph_borders(title)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(110)
    sr = subtitle.add_run("Итоговый отчёт по проекту LawanKZ")
    sr.font.size = Pt(16)
    sr.font.name = "Times New Roman"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Тематический корпус нормативных правовых актов Республики Казахстан").font.size = Pt(14)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(110)
    p.add_run("2026").font.size = Pt(14)

    # Содержание
    doc.add_page_break()
    doc.add_heading("Содержание", level=1)
    toc = doc.add_paragraph()
    add_field(toc, 'TOC \\o "1-1" \\h \\z \\u', "Содержание будет обновлено при открытии документа")
    add_body(
        doc,
        "Структура отчёта следует фактическому ходу проекта: от постановки задачи и формирования корпуса до обучения эмбеддингов и проверки семантического поиска. Приложение содержит сокращённые листинги ключевых функций; полный код хранится в репозитории.",
    )

    # Аннотация и введение
    chapter(doc, "Аннотация", page_break=True)
    add_body(
        doc,
        "В работе разработан воспроизводимый конвейер анализа русскоязычных нормативных правовых актов Республики Казахстан. Исходный корпус включает шесть актов цифровой тематики общим объёмом 62 467 слов. Тексты получены из информационно правовой системы Әділет, сохранены локально и разделены на 325 статей. Предобработка сочетает токенизацию и морфологический анализ spaCy, удаление пунктуации и общих стоп слов, лемматизацию и стемминг Snowball. Для статистического представления построены матрицы Bag of Words и TF IDF размером 325 на 1718 признаков. Для распределённого представления обучены Word2Vec, fastText и учебная модель GloVe с размерностью векторов 100. Прикладная часть представляет собой поиск похожих статей по косинусному сходству средних Word2Vec векторов с весами IDF. В автоматической leave one out проверке получены precision@5 0,6566 и hit rate@5 0,8831. Результаты показывают, что небольшой тематический корпус достаточен для демонстрационного поиска, но не для юридически значимых заключений без ручной экспертной оценки.",
    )
    p = doc.add_paragraph()
    p.style = doc.styles["Body Text"]
    p.paragraph_format.first_line_indent = Cm(0)
    p.add_run("Ключевые слова ").bold = True
    p.add_run("обработка естественного языка, нормативный правовой акт, лемматизация, TF IDF, Word2Vec, GloVe, fastText, семантический поиск")

    doc.add_heading("1 Введение", level=1)
    add_body(
        doc,
        "Нормативные правовые акты образуют массив текстов со сложной структурой и специальной лексикой. В одном документе повторяются названия органов, ссылки на другие положения, определения и модальные конструкции. Ручной просмотр остаётся обязательным для юридического толкования, однако методы обработки естественного языка помогают организовать корпус, выявить частотную лексику и предварительно найти тематически близкие статьи.",
    )
    add_body(
        doc,
        "Проект LawanKZ исследует, насколько классические методы NLP применимы к небольшому русскоязычному корпусу законодательства цифровой сферы. Работа выполнялась последовательно в течение семи учебных недель. Первые четыре этапа охватывали корпус, очистку, морфологию, Bag of Words и TF IDF. На пятом этапе были обучены распределённые представления слов. На шестом этапе реализован поиск похожих статей. Седьмой этап объединяет код, численные результаты и ограничения в итоговый отчёт.",
    )
    add_body(
        doc,
        "Практический результат проекта состоит в командном прототипе: пользователь вводит запрос на русском языке, система обрабатывает его тем же конвейером, что и корпус, строит вектор запроса и возвращает статьи с наибольшим косинусным сходством. Прототип не отвечает на правовые вопросы и не определяет действие нормы. Его назначение ограничено навигацией по подготовленной выборке.",
    )

    # Постановка задачи
    chapter(doc, "2 Постановка задачи и организация исследования")
    doc.add_heading("2 1 Цель и задачи", level=2)
    add_body(
        doc,
        "Цель работы состоит в разработке и проверке воспроизводимого конвейера анализа русскоязычных НПА, который объединяет лингвистическую обработку, статистические признаки, словесные эмбеддинги и поиск тематически похожих статей.",
    )
    add_bullets(
        doc,
        [
            "сформировать тематический корпус объёмом не менее 50 000 слов и зафиксировать источники каждого документа",
            "реализовать очистку текста с сохранением юридически значимых отрицаний и модальных слов",
            "сравнить словоформы, леммы spaCy и стемы Snowball, а также получить морфологическую статистику",
            "построить и интерпретировать матрицы Bag of Words и TF IDF",
            "обучить Word2Vec, GloVe и fastText на одних данных и сравнить их поведение",
            "создать прототип поиска похожих статей и оценить его на автоматической разметке",
        ],
    )
    doc.add_heading("2 2 Объект предмет и гипотеза", level=2)
    add_body(
        doc,
        "Объект исследования — русскоязычные тексты нормативных правовых актов Республики Казахстан, относящиеся к цифровому регулированию. Предмет исследования — языковые признаки и векторные представления, которые позволяют сравнивать статьи этих актов. Рабочая гипотеза заключается в том, что сочетание лемматизации, IDF весов и обученных на корпусе Word2Vec векторов позволит поднять тематически релевантные статьи в верхнюю часть выдачи даже без сложной нейросетевой модели.",
    )
    doc.add_heading("2 3 Критерии проверки", level=2)
    add_body(
        doc,
        "Корректность конвейера проверялась на трёх уровнях. Структурные тесты контролировали извлечение текста и сохранение идентификаторов статей. Лингвистические тесты проверяли, что отрицание не и слова должен, данные, без не исчезают как стоп слова. Итоговая проверка поиска измеряла долю соседей из того же исходного акта в первых пяти позициях. Такая оценка упрощает понятие релевантности, поэтому её значение рассматривается как технический ориентир, а не как доказательство юридической пригодности.",
    )
    add_figure(doc, pipeline_path, 6.25, "Рисунок 1  Общая последовательность обработки корпуса")

    # Теория
    chapter(doc, "3 Теоретические основы")
    doc.add_heading("3 1 Нормализация текста", level=2)
    add_body(
        doc,
        "Токенизация делит строку на отдельные единицы, после чего к каждому токену применяются фильтры. Для русского языка одного перевода в нижний регистр недостаточно: слова закон, закона и закону необходимо связать с общей словарной формой. В проекте эту функцию выполняет лемматизатор русской модели spaCy. Стеммер Snowball используется для сравнения и обрезает слово до условной основы. Лемма обычно понятна человеку, тогда как стем может не совпадать с реальным словом.",
    )
    add_body(
        doc,
        "Стандартные списки стоп слов требуют адаптации к правовой области. Удаление частицы не способно обратить смысл обязанности или запрета. По этой причине проект вводит отдельное множество сохраняемых слов. В него входят отрицания, слова должен, обязан, вправе, запрещается, подлежит и термин данные. Такое решение не устраняет все смысловые ошибки, но предотвращает наиболее очевидное искажение нормы на этапе очистки.",
    )
    doc.add_heading("3 2 Статистические представления", level=2)
    add_body(
        doc,
        "Bag of Words представляет текст количеством употреблений каждого термина. Метод прозрачен и удобен для частотного анализа, но не учитывает порядок слов и снижает различимость документов с общей канцелярской лексикой. TF IDF дополняет частоту обратной документной частотой. Термин получает больший вес, если часто встречается в конкретной статье и сравнительно редко встречается в других статьях. В реализации использованы CountVectorizer и TfidfTransformer библиотеки scikit learn [4; 8].",
    )
    doc.add_heading("3 3 Распределённые представления", level=2)
    add_body(
        doc,
        "Word2Vec обучает плотные векторы на локальных контекстах. В варианте skip gram модель по центральному слову предсказывает окружающие слова, вследствие чего единицы с похожим употреблением сближаются в векторном пространстве [1]. GloVe использует глобальные количества совместной встречаемости и оптимизирует взвешенную регрессионную цель по ненулевым элементам матрицы контекстов [2]. fastText расширяет skip gram символьными n граммами и может построить вектор для формы, отсутствующей в словаре [3].",
    )
    add_body(
        doc,
        "Сходство двух нормированных векторов оценивается их скалярным произведением, эквивалентным косинусной мере. Значение близкое к единице означает сходное направление векторов, но не юридическую эквивалентность текстов. Контекстная близость может отражать общую тему, типовую формулировку или регулярное соседство терминов.",
    )

    # Корпус
    chapter(doc, "4 Корпус и воспроизводимость данных")
    doc.add_heading("4 1 Состав корпуса", level=2)
    add_body(
        doc,
        "Корпус собран из официальных страниц информационно правовой системы Әділет. Выбраны шесть актов цифровой тематики. Такой состав создаёт связную предметную область, но одновременно усиливает общую лексику: слова цифровой, информация, объект и орган встречаются в нескольких документах. Суммарный объём 62 467 слов удовлетворяет альтернативному условию учебного задания в 50 000 слов. Число документов остаётся малым, поэтому выводы нельзя переносить на всё законодательство Казахстана.",
    )
    corpus_rows = [
        [str(index), item["title"], item["topic"], f"{item['word_count']:,}".replace(",", " ")]
        for index, item in enumerate(corpus["documents"], start=1)
    ]
    add_table(doc, ["№", "Нормативный правовой акт", "Тематика", "Слов"], corpus_rows, [0.35, 3.1, 1.8, 0.7])
    p = doc.add_paragraph("Таблица 1  Состав исследуемого корпуса", style="Caption")
    doc.add_heading("4 2 Получение и фиксация", level=2)
    add_body(
        doc,
        "Скрипт build_corpus.py читает манифест источников, проверяет правила robots.txt, загружает HTML и сохраняет его вместе со временем получения и контрольной суммой SHA 256. При последующих запусках используются локальные копии. Это важно для воспроизводимости: официальный текст может быть отредактирован позднее, а сохранённая страница фиксирует фактический вход эксперимента на 22 сентября 2026 года.",
    )
    add_body(
        doc,
        "Из HTML извлекается содержимое контейнера действующего акта. Навигация, скрипты, стили и редакционные примечания исключаются. Затем цельный текст разделяется по заголовкам статей. Из 6 документов сформировано 325 статей длиной не менее десяти слов. Именно статья служит строкой матриц и единицей поиска; исходный акт остаётся метаданными для анализа и оценки.",
    )
    doc.add_heading("4 3 Организация файлов", level=2)
    add_body(
        doc,
        "Каталог data raw содержит исходные HTML страницы и метаданные загрузки. В data processed находятся JSONL корпус и сводная статистика. Результаты недель 1–4 записываются в results weeks_1_4, а модели и поиск — в results weeks_5_6. Разделение исходных данных, обработанных текстов и производных артефактов облегчает повторный запуск и проверку отдельных этапов.",
    )

    # Предобработка
    chapter(doc, "5 Предобработка и морфологический анализ")
    doc.add_heading("5 1 Конвейер очистки", level=2)
    add_body(
        doc,
        "Функция preprocess_text принимает произвольную строку и возвращает три синхронных списка: очищенные словоформы, леммы и стемы. Остаются русские последовательности длиной не менее двух символов. Числа и пунктуация исключаются. Стоп слова проверяются одновременно для исходной формы и леммы, после чего применяется доменное исключение KEEP_WORDS.",
    )
    add_code(
        doc,
        '''def preprocess_text(text: str, nlp=None):
    nlp = nlp or spacy.load("ru_core_news_sm", disable=["ner", "parser"])
    stemmer = SnowballStemmer("russian")
    words, lemmas, stems = [], [], []
    for token in nlp(text):
        cleaned = clean_token(token)
        if cleaned is None:
            continue
        word, lemma = cleaned
        words.append(word)
        lemmas.append(lemma)
        stems.append(stemmer.stem(word))
    return {"words": words, "lemmas": lemmas, "stems": stems}''',
        "Листинг 1  Сокращённая функция предобработки",
    )
    add_body(
        doc,
        "На примере предложения Оператор не должен передавать персональные данные без согласия субъекта пунктуация удаляется, однако не, должен и без сохраняются. В полном корпусе до фильтрации насчитано 60 004 словесных токена, после фильтрации — 45 924. Показатель 62 467 относится ко всему извлечённому тексту и простому подсчёту слов, поэтому он не обязан совпадать с количеством токенов spaCy в отобранных статьях.",
    )
    doc.add_heading("5 2 Леммы и стемы", level=2)
    add_body(
        doc,
        "После очистки получено 6 374 уникальные словоформы и 2 838 уникальных лемм. Лемматизация объединяет формы информации и информацией с леммой информация, а республики и республике — с леммой республика. Snowball выдаёт основы информац и республик. Стем полезен для грубого сопоставления, но хуже подходит для объяснения результата пользователю.",
    )
    add_body(
        doc,
        "spaCy также определяет часть речи, падеж и число. Эти признаки сохранены в morphology.csv и дают описание грамматического состава корпуса. Модель обучалась на общем русском языке и может ошибаться на терминах. Например, отдельные формы слова кибербезопасность нормализуются непоследовательно. Поэтому морфологическая таблица рассматривается как автоматическая разметка, а не как эталон.",
    )
    add_figure(doc, RESULTS_1_4 / "wordcloud_lemmas.png", 6.2, "Рисунок 2  Облако наиболее частотных лемм корпуса")

    # BoW TFIDF
    chapter(doc, "6 Представления Bag of Words и TF IDF")
    doc.add_heading("6 1 Построение матриц", level=2)
    add_body(
        doc,
        "Для каждой статьи леммы соединяются в строку. CountVectorizer формирует словарь только из терминов, встреченных минимум в двух статьях; термины, присутствующие более чем в 95 процентах статей, исключаются. На счётчиках обучается TfidfTransformer. Обе матрицы имеют 325 строк и 1718 столбцов, поэтому значение в одинаковой позиции относится к одной статье и одной лемме.",
    )
    add_code(
        doc,
        '''texts = [" ".join(article["lemmas"]) for article in cleaned_articles]
vectorizer = CountVectorizer(min_df=2, max_df=0.95)
bow = vectorizer.fit_transform(texts)
tfidf = TfidfTransformer().fit_transform(bow)
features = vectorizer.get_feature_names_out()''',
        "Листинг 2  Построение BoW и TF IDF",
    )
    doc.add_heading("6 2 Сравнение результатов", level=2)
    add_body(
        doc,
        "По суммарной частоте BoW лидируют леммы цифровой, казахстан, республика, связь и орган. Средний TF IDF также сохраняет часть этой лексики, поскольку корпус тематически узок, а статьи различаются по длине. Это наблюдение показывает, что TF IDF не превращает частотный список в автоматический перечень юридически значимых понятий. Метод меняет веса относительно распределения терминов по статьям.",
    )
    add_body(
        doc,
        "Для более содержательной интерпретации рассчитана разность между средним TF IDF статей конкретного акта и средним по остальным актам. У закона о персональных данных выделяются леммы персональный, сбор и обработка. У акта об искусственном интеллекте — искусственный, интеллект и система. У закона о связи — связь, сеть и устройство. Контраст соответствует тематике документов и подтверждает, что матрица сохраняет предметные различия.",
    )
    add_figure(doc, RESULTS_1_4 / "bow_vs_tfidf.png", 6.3, "Рисунок 3  Сопоставление частот BoW и средних весов TF IDF")

    # Embeddings
    chapter(doc, "7 Обучение и сравнение эмбеддингов")
    doc.add_heading("7 1 Параметры обучения", level=2)
    add_body(
        doc,
        "Word2Vec и fastText обучены средствами Gensim на списках лемм 325 статей. Для обеих моделей выбраны размерность 100, окно 5, минимальная частота 3, skip gram, 10 отрицательных примеров и 80 эпох. Число потоков ограничено одним, seed равен 42. Эти настройки повышают воспроизводимость и дают модели несколько проходов по малому корпусу.",
    )
    add_body(
        doc,
        "GloVe реализован в проекте на NumPy, чтобы сравнение не зависело от готовых англоязычных векторов. Матрица совместной встречаемости использует окно 5 и вес, обратно пропорциональный расстоянию между словами. Оптимизация выполнялась 35 эпох. Среднее значение функции потерь снизилось с 0,0383 до 0,0111. Снижение показывает сходимость оптимизации, но не гарантирует высокое качество семантики.",
    )
    add_code(
        doc,
        '''common = dict(sentences=sentences, vector_size=100, window=5,
              min_count=3, workers=1, epochs=80, seed=42)
word2vec = Word2Vec(sg=1, negative=10, **common)
fasttext = FastText(sg=1, negative=10, min_n=3, max_n=6,
                    bucket=50_000, **common)
glove, losses = train_glove(sentences, vector_size=100, window=5,
                            min_count=3, epochs=35)''',
        "Листинг 3  Единые параметры обучения эмбеддингов",
    )
    doc.add_heading("7 2 Семантические примеры", level=2)
    add_body(
        doc,
        "Ближайшие соседи Word2Vec отражают регулярные контексты корпуса. Для слова связь первыми получены оператор, сеть, сотовый, эксплуатировать и услуга. Для интеллект — искусственный, модель, система, синтетический и способность. Для кибербезопасности — обеспечение, координационный, центр, сфера и исследователь. Эти результаты читаются как тематические ассоциации, а не как синонимы.",
    )
    add_body(
        doc,
        "Векторная арифметика оказалась менее устойчивой. Выражение цифровой минус среда плюс система у Word2Vec возвращает объект, искусственный и интеллект, но другие комбинации дают слабые связи. На 45 924 токенах недостаточно повторяющихся отношений для стабильных линейных аналогий. Этот отрицательный результат важен: арифметика векторов демонстрирует геометрию модели, но не реализует логический вывод.",
    )

    # Comparison
    chapter(doc, "8 Оценка моделей слов")
    add_body(
        doc,
        "Для внутреннего сравнения создан небольшой диагностический набор из пяти тематических групп и шести связанных пар. Все проверочные слова присутствуют в словарях. Средний косинус измеряет близость пар персональный и данные, искусственный и интеллект, электронный и подпись, оператор и связь, субъект и согласие, угроза и безопасность. Тематическая precision@5 показывает долю слов той же ручной группы среди пяти соседей.",
    )
    model_rows = [
        [
            item["model"],
            str(item["vocabulary_size"]),
            f"{item['mean_association_cosine']:.4f}".replace(".", ","),
            f"{item['mean_group_precision_at_5']:.4f}".replace(".", ","),
            "да" if item["oov_typo_supported"] else "нет",
        ]
        for item in summary_5_6["models"]
    ]
    add_table(
        doc,
        ["Модель", "Словарь", "Косинус пар", "Precision@5 групп", "Вектор опечатки"],
        model_rows,
        [1.15, 0.8, 1.2, 1.35, 1.25],
    )
    doc.add_paragraph("Таблица 2  Сравнение моделей на диагностическом наборе", style="Caption")
    add_body(
        doc,
        "GloVe показал наибольшие значения двух внутренних метрик: 0,7334 для ассоциативных пар и 0,2714 для тематических соседей. Это не доказывает общего превосходства GloVe, поскольку набор составлен для данного проекта и остаётся малым. fastText получил близкий к Word2Vec средний косинус, но поддержал отсутствующую форму персональнвй за счёт символьных n грамм. Одновременно его соседи иногда оказываются только морфологическими вариантами слова, что снижает практическую ценность результата.",
    )
    add_body(
        doc,
        "Для прикладного поиска выбран Word2Vec. Такое решение не связано с лучшим числом в таблице: Word2Vec даёт понятные соседства, имеет простой механизм загрузки и не создаёт вектор для произвольной ошибочной строки. GloVe сохранён для сравнения, а fastText может быть полезен в будущей версии с пользовательскими опечатками и нерегулярной терминологией.",
    )

    # Search
    chapter(doc, "9 Семантический поиск похожих статей")
    doc.add_heading("9 1 Вектор статьи и запрос", level=2)
    add_body(
        doc,
        "Вектор статьи вычисляется как среднее Word2Vec векторов известных модели лемм. Каждая лемма умножается на свой IDF, восстановленный из матрицы BoW. Частые во многих статьях слова влияют слабее, а более специфичные — сильнее. Полученный вектор нормируется. Запрос обрабатывается той же функцией preprocess_text и тем же набором весов, поэтому его можно непосредственно сравнить со статьями.",
    )
    add_code(
        doc,
        '''query_vector = weighted_mean_vector(query_words, model, idf)
if query_vector is None:
    raise ValueError("В запросе нет слов, известных модели")
similarities = article_vectors @ query_vector
for index in np.argsort(-similarities)[:top_k]:
    results.append({
        "article_id": articles[index]["article_id"],
        "similarity": float(similarities[index])
    })''',
        "Листинг 4  Ранжирование статей по косинусному сходству",
    )
    doc.add_heading("9 2 Примеры выдачи", level=2)
    first_query = samples[0]
    search_rows = []
    for rank, result in enumerate(first_query["results"][:5], start=1):
        search_rows.append(
            [
                str(rank),
                result["document_title"],
                str(result["article_number"]),
                result["article_title"],
                f"{result['similarity']:.4f}".replace(".", ","),
            ]
        )
    add_body(doc, f"Для запроса «{first_query['query']}» система получила следующие первые результаты.")
    add_table(doc, ["№", "Акт", "Статья", "Заголовок", "Сходство"], search_rows, [0.32, 2.05, 0.75, 2.4, 0.7])
    doc.add_paragraph("Таблица 3  Первые результаты семантического поиска", style="Caption")
    add_body(
        doc,
        "Первые позиции относятся к закону о персональных данных: негосударственный сервис, порядок согласия на сбор и обработку, права субъекта, распространение данных и определения. Другие подготовленные запросы также дали тематически согласованную выдачу. Фраза об угрозах кибербезопасности вывела Центр обеспечения кибербезопасности и отраслевой центр. Запрос о рисках ИИ поставил первой статью об управлении рисками систем искусственного интеллекта. Запрос об электронной цифровой подписи вывел одноимённую статью Цифрового кодекса.",
    )

    # Evaluation and PCA
    chapter(doc, "10 Оценка поиска и визуализация")
    doc.add_heading("10 1 Автоматическая проверка", level=2)
    add_body(
        doc,
        "Для количественной проверки каждая статья использовалась как запрос к остальным 324 статьям. Сама статья исключалась из ранжирования. Сосед считался релевантным, если относился к тому же исходному акту. Средняя precision@5 составила 0,6566: примерно две трети первых пяти соседей происходят из того же документа. Hit rate@5 равна 0,8831: хотя бы один такой сосед найден для 88,31 процента статей.",
    )
    metrics_rows = [
        ["Количество статей", str(summary_5_6["retrieval"]["article_count"])],
        ["Глубина выдачи", str(summary_5_6["retrieval"]["top_k"])],
        ["Precision@5", str(summary_5_6["retrieval"]["precision_at_5"]).replace(".", ",")],
        ["Hit rate@5", str(summary_5_6["retrieval"]["hit_rate_at_5"]).replace(".", ",")],
    ]
    add_table(doc, ["Показатель", "Значение"], metrics_rows, [3.8, 1.6])
    doc.add_paragraph("Таблица 4  Показатели автоматической оценки поиска", style="Caption")
    add_body(
        doc,
        "Такая разметка не совпадает с пользовательской релевантностью. Статьи разных актов могут регулировать близкий вопрос, а статьи одного закона — описывать разные процедуры. Поэтому метрики подтверждают тематическую группировку, но не заменяют набор запросов с ручными оценками. Для учебного прототипа они дают воспроизводимую точку отсчёта.",
    )
    doc.add_heading("10 2 Проекция PCA", level=2)
    add_body(
        doc,
        "Для визуализации 100 мерные векторы статей преобразованы методом главных компонент в две координаты [9]. Диаграмма показывает заметную область статей о связи в нижней левой части, персональные данные в верхней левой части и искусственный интеллект справа. Статьи Цифрового кодекса распределены шире и пересекаются с другими актами, что соответствует его комплексной тематике.",
    )
    add_figure(doc, RESULTS_5_6 / "article_vectors_pca.png", 6.35, "Рисунок 4  PCA проекция векторов статей")
    add_body(
        doc,
        "Две компоненты неизбежно теряют значительную часть информации исходного пространства. Расстояние на рисунке нельзя использовать вместо косинусного сходства 100 мерных векторов. PCA служит пояснением общей структуры и не участвует в поисковом ранжировании.",
    )

    # Implementation
    chapter(doc, "11 Программная реализация и проверка")
    doc.add_heading("11 1 Технологический стек", level=2)
    stack_rows = [
        ["Python 3.13", "основной язык и запуск сценариев"],
        ["Beautiful Soup и HTTPX", "получение и разбор официальных HTML страниц"],
        ["spaCy ru_core_news_sm", "токенизация, лемматизация и морфологические признаки"],
        ["NLTK Snowball", "русский стемминг для сравнения с леммами"],
        ["scikit learn", "BoW, TF IDF, PCA и вспомогательные преобразования"],
        ["Gensim", "обучение и сохранение Word2Vec и fastText"],
        ["NumPy и SciPy", "GloVe, разреженные матрицы и векторные вычисления"],
        ["matplotlib и WordCloud", "графики и облако лемм"],
    ]
    add_table(doc, ["Средство", "Назначение"], stack_rows, [2.0, 4.0])
    doc.add_paragraph("Таблица 5  Основные программные средства", style="Caption")
    doc.add_heading("11 2 Воспроизводимый запуск", level=2)
    add_body(
        doc,
        "Зависимости зафиксированы в requirements-week4.txt, включая прямую ссылку на русскую модель spaCy. Скрипт build_corpus.py создаёт корпус, run_weeks_1_4.py пересчитывает лингвистические таблицы и разреженные матрицы, run_weeks_5_6.py обучает эмбеддинги и строит поиск. Ноутбуки weeks_1_4.ipynb и weeks_5_6.ipynb исполнены целиком и содержат результаты без ошибок.",
    )
    add_code(
        doc,
        '''.\\.venv\\Scripts\\python.exe .\\scripts\\build_corpus.py
.\\.venv\\Scripts\\python.exe .\\scripts\\run_weeks_1_4.py
.\\.venv\\Scripts\\python.exe .\\scripts\\run_weeks_5_6.py
.\\.venv\\Scripts\\python.exe .\\scripts\\search_similar.py \
  "риски системы искусственного интеллекта" --top 5''',
        "Листинг 5  Последовательность запуска в PowerShell",
    )
    doc.add_heading("11 3 Автоматические тесты", level=2)
    add_body(
        doc,
        "Набор из 16 unit тестов выполняется без сетевых запросов. Проверяются извлечение текста без навигации и примечаний, отказ от страниц без контейнера действующего акта, сохранение номера и источника статьи, предобработка отрицаний, преобразование JSON в Markdown, нормализация заголовков и базовые операции векторного поиска. Тесты не измеряют юридическую корректность, но защищают воспроизводимость программных преобразований.",
    )

    # Results and limitations
    chapter(doc, "12 Обсуждение результатов")
    doc.add_heading("12 1 Основные наблюдения", level=2)
    add_body(
        doc,
        "Лемматизация уменьшила число уникальных единиц с 6 374 словоформ до 2 838 лемм. Это сокращение особенно важно для русского языка с развитой флексией: статистические матрицы становятся компактнее, а частоты грамматических форм объединяются. Одновременно качество результата зависит от лемматизатора, который ошибается на специализированных терминах.",
    )
    add_body(
        doc,
        "BoW и TF IDF показали две стороны корпуса. Частотная модель хорошо отражает доминирующую лексику, но поднимает общие слова цифровой, казахстан и республика. Контрастные TF IDF признаки лучше отделяют темы отдельных актов. Однако средний TF IDF по всем статьям не устраняет общую лексику автоматически, что необходимо объяснять при интерпретации графика.",
    )
    add_body(
        doc,
        "Эмбеддинги обнаружили устойчивые пары искусственный и интеллект, связь и оператор, персональный и данные. GloVe получил лучшие показатели на небольшом внутреннем наборе, fastText поддержал слово с опечаткой, а Word2Vec дал прозрачную основу для статьи. Различия моделей подтверждают, что выбор представления должен учитывать прикладную задачу, а не одну усреднённую метрику.",
    )
    add_body(
        doc,
        "Поиск продемонстрировал практическую ценность в пяти контрольных запросах. В каждом случае верхние позиции соответствовали ожидаемому акту и конкретным статьям. Автоматические метрики также выше случайного тематического совпадения. Тем не менее прототип ранжирует тексты по употреблению слов, а не устанавливает правовые связи, и не учитывает редакции, иерархию актов или ссылки между нормами.",
    )
    doc.add_heading("12 2 Ограничения достоверности", level=2)
    add_bullets(
        doc,
        [
            "корпус содержит только шесть актов одной широкой тематики и не представляет все виды НПА",
            "единица анализа статья создаёт 325 строк, но они зависимы внутри одного исходного документа",
            "морфологическая модель общего русского языка не гарантирует точную разметку юридической терминологии",
            "внутренняя оценка эмбеддингов использует малый авторский набор понятий",
            "релевантность поиска приближённо определяется принадлежностью к одному акту, а не оценками юриста",
            "сохранённые страницы фиксируют редакции на дату загрузки и не подтверждают актуальность нормы в будущем",
        ],
    )
    doc.add_heading("12 3 Направления развития", level=2)
    add_body(
        doc,
        "Следующая версия может расширить корпус, добавить метаданные об органе, дате и статусе акта, а также построить ручной набор запросов с оценками релевантности. Для поиска целесообразно сравнить текущий Word2Vec IDF подход с BM25 и многоязычными трансформерными эмбеддингами. Отдельное направление — извлечение ссылок между статьями и поиск модальных конструкций с учётом отрицания и субъекта обязанности.",
    )

    # Conclusion
    chapter(doc, "13 Заключение")
    add_body(
        doc,
        "В проекте создан сквозной воспроизводимый конвейер анализа русскоязычных нормативных правовых актов. Корпус из шести актов и 62 467 слов сохранён вместе с метаданными и контрольными суммами. Тексты разделены на 325 статей, очищены и лемматизированы. Построены матрицы Bag of Words и TF IDF размером 325 на 1718, частотные таблицы и визуализации.",
    )
    add_body(
        doc,
        "На подготовленных леммах обучены Word2Vec, fastText и GloVe. Сравнение показало понятные тематические соседства и одновременно выявило нестабильность векторной арифметики на малом корпусе. На основе Word2Vec и IDF разработан поиск похожих статей. Его автоматическая precision@5 равна 0,6566, hit rate@5 — 0,8831. Пять контрольных запросов выводят ожидаемые статьи о персональных данных, кибербезопасности, связи, искусственном интеллекте и электронной подписи.",
    )
    add_body(
        doc,
        "Рабочая гипотеза подтверждена в границах учебного эксперимента: классические методы NLP достаточны для тематической навигации по небольшому корпусу. Результат нельзя использовать для юридической консультации или определения применимости нормы. Практическая ценность прототипа состоит в прозрачности вычислений, воспроизводимости и возможности заменить отдельные компоненты при дальнейшем расширении проекта.",
    )

    # Bibliography
    chapter(doc, "Список использованных источников", page_break=True)
    references = [
        ("1", "Mikolov T., Chen K., Corrado G., Dean J. Efficient Estimation of Word Representations in Vector Space. 2013.", "https://arxiv.org/abs/1301.3781"),
        ("2", "Pennington J., Socher R., Manning C. GloVe: Global Vectors for Word Representation. EMNLP. 2014. P. 1532–1543. DOI 10.3115/v1/D14-1162.", "https://aclanthology.org/D14-1162/"),
        ("3", "Bojanowski P., Grave E., Joulin A., Mikolov T. Enriching Word Vectors with Subword Information. TACL. 2017. Vol. 5. P. 135–146. DOI 10.1162/tacl_a_00051.", "https://aclanthology.org/Q17-1010/"),
        ("4", "Salton G., Buckley C. Term-weighting approaches in automatic text retrieval. Information Processing and Management. 1988. Vol. 24. No. 5. P. 513–523." , "https://doi.org/10.1016/0306-4573(88)90021-0"),
        ("5", "Honnibal M., Montani I., Van Landeghem S., Boyd A. spaCy Industrial-strength Natural Language Processing in Python. 2020.", "https://spacy.io/"),
        ("6", "Bird S., Klein E., Loper E. Natural Language Processing with Python. O’Reilly Media, 2009.", "https://www.nltk.org/book/"),
        ("7", "Řehůřek R., Sojka P. Software Framework for Topic Modelling with Large Corpora. LREC Workshop. 2010. P. 45–50.", "https://radimrehurek.com/gensim/"),
        ("8", "Pedregosa F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011. Vol. 12. P. 2825–2830.", "https://jmlr.org/papers/v12/pedregosa11a.html"),
        ("9", "Jolliffe I. T., Cadima J. Principal component analysis: a review and recent developments. Philosophical Transactions A. 2016. Vol. 374.", "https://doi.org/10.1098/rsta.2015.0202"),
        ("10", "Закон Республики Казахстан О персональных данных и их защите. ИПС Әділет.", "https://old.adilet.zan.kz/rus/docs/Z1300000094"),
        ("11", "Закон Республики Казахстан О кибербезопасности. ИПС Әділет.", "https://old.adilet.zan.kz/rus/docs/Z1500000418"),
        ("12", "Цифровой кодекс Республики Казахстан. ИПС Әділет.", "https://old.adilet.zan.kz/rus/docs/K2600000255"),
        ("13", "Закон Республики Казахстан О связи. ИПС Әділет.", "https://old.adilet.zan.kz/rus/docs/Z040000567_"),
        ("14", "Закон Республики Казахстан О доступе к информации. ИПС Әділет.", "https://old.adilet.zan.kz/rus/docs/Z1500000401"),
        ("15", "Закон Республики Казахстан Об искусственном интеллекте. ИПС Әділет.", "https://old.adilet.zan.kz/rus/docs/Z2500000230"),
    ]
    for number, citation, url in references:
        p = doc.add_paragraph()
        p.style = doc.styles["Body Text"]
        p.paragraph_format.first_line_indent = Cm(-0.75)
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        p.add_run(f"{number}. {citation} ")
        add_hyperlink(p, url, url)

    # Appendix
    chapter(doc, "Приложение А Структура результатов", page_break=True)
    add_body(
        doc,
        "Ниже перечислены основные файлы, которые позволяют проверить выводы отчёта без повторного обучения моделей.",
    )
    artifact_rows = [
        ["documents.jsonl", "тексты шести актов и метаданные источников"],
        ["processed_articles.jsonl", "325 статей с очищенными словами, леммами и стемами"],
        ["top50_words.csv и top50_lemmas.csv", "частотные списки недели 3"],
        ["bow_matrix.npz и tfidf_matrix.npz", "разреженные матрицы размером 325 на 1718"],
        ["word2vec.model", "модель Word2Vec для поиска"],
        ["fasttext.model и glove.kv", "сравниваемые модели слов"],
        ["sample_search_results.json", "пять контрольных запросов и первые результаты"],
        ["retrieval_metrics.json", "precision@5 и hit rate@5"],
    ]
    add_table(doc, ["Файл", "Содержание"], artifact_rows, [2.7, 3.3])
    doc.add_paragraph("Таблица А 1  Артефакты вычислительного эксперимента", style="Caption")
    doc.add_heading("Команды проверки", level=2)
    add_code(
        doc,
        '''# Полный набор автоматических тестов
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v

# Открытие двух исполненных ноутбуков
.\\.venv\\Scripts\\python.exe -m jupyter lab .\\notebooks''',
        "Листинг А 1  Проверка тестов и открытие ноутбуков",
    )
    add_body(
        doc,
        "Все численные значения в отчёте читаются из файлов результатов при сборке DOCX. Такой подход снижает риск расхождения между текстом и текущим состоянием проекта. Если модели будут переобучены с другими параметрами или корпус изменится, отчёт следует собрать заново.",
    )

    doc.save(REPORT_PATH)
    return REPORT_PATH


if __name__ == "__main__":
    path = build_report()
    print(path)
