
# upload_pdf.py

import gradio as gr
import PyPDF2

def parse_pdf(file_obj):
    """
    解析上传的PDF文件，提取文本内容。
    参数:
        file_obj: Gradio上传的文件对象。
    返回:
        text_content: 提取的文本内容。
    """
    if file_obj is None:
        return "No file uploaded."
    
    if not file_obj.name.endswith('.pdf'):  
        return "请上传一个 PDF 文件。"

    # PyPDF2 expects a file-like object, Gradio provides a tempfile._TemporaryFileWrapper
    pdf_reader = PyPDF2.PdfFileReader(file_obj)
    text_content = ''
    num_pages = pdf_reader.getNumPages()
    for page_num in range(num_pages):
        page = pdf_reader.getPage(page_num)
        text_content += page.extractText()
    
    return text_content[:100]

# 创建Gradio界面
iface = gr.Interface(
    fn=parse_pdf,
    inputs=gr.File(),  # 设置只允许上传 PDF 文件  
    outputs="text",
    title="PDF Upload and Parse",
    description="Upload a PDF file and extract its text content."
)

if __name__ == "__main__":
    iface.launch(share=True)


