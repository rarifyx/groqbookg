import streamlit as st
import json
import os
from io import BytesIO
from md2pdf.core import md2pdf
from google.generativeai import GenerationConfig, GenerativeModel

# Replace these with your API key and model name
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
MODEL_NAME = "gemini-pro"  # Choose from available models on Google AI Platform

if 'api_key' not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY

if 'model' not in st.session_state:
    if GEMINI_API_KEY:
        st.session_state.model = GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GenerationConfig(
                temperature=1,  # Adjust these as needed
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                response_mime_type="text/plain",
            ),
        )

class GenerationStatistics:
    def __init__(self, input_time=0,output_time=0,input_tokens=0,output_tokens=0,total_time=0,model_name="gemini-pro"):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time # Sum of queue, prompt (input), and completion (output) times
        self.model_name = model_name

    def get_input_speed(self):
        """ 
        Tokens per second calculation for input
        """
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0
    
    def get_output_speed(self):
        """ 
        Tokens per second calculation for output
        """
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0
    
    def add(self, other):
        """
        Add statistics from another GenerationStatistics object to this one.
        """
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")
        
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        return (f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n\n"
                f"| Metric          | Input          | Output          | Total          |\n"
                f"|-----------------|----------------|-----------------|----------------|\n"
                f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(self.input_tokens + self.output_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
                f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {self.input_tokens + self.output_tokens}            |\n"
                f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |")

class Book:
    def __init__(self, structure):
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}

        st.markdown("## Generating the following:")
        toc_columns = st.columns(4)
        self.display_toc(self.structure, toc_columns)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        
        for title, content in structure.items():
            if self.contents[title].strip():  # Only display title if there is content
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        """
        Returns the markdown styled pure string with the contents.
        """
        if structure is None:
            structure = self.structure
        
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():  # Only include title if there is content
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

def create_markdown_file(content: str) -> BytesIO:
    """
    Create a Markdown file from the provided content.
    """
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

def create_pdf_file(content: str):
    """
    Create a PDF file from the provided content.
    """
    pdf_buffer = BytesIO()
    md2pdf(pdf_buffer, md_content=content)
    pdf_buffer.seek(0)
    return pdf_buffer


def generate_book_structure(prompt: str):
    """
    Generates the book structure using Gemini.
    """
    if not st.session_state.api_key:
        raise ValueError("Please provide a valid Gemini API key.")

    response = st.session_state.model.predict(
        prompt=f"Write in JSON format:\n\n{{\"Title of section goes here\":\"Description of section goes here\",\n\"Title of section goes here\":{{\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\"}}}}",
        temperature=0.5,  # Adjust temperature for creativity
        max_output_tokens=8000,
    )

    try:
        book_structure_json = json.loads(response.text)
        return book_structure_json
    except json.JSONDecodeError:
        st.error("Failed to decode the book structure. Please try again.")

def generate_section(prompt: str):
    """
    Generates a section using Gemini.
    """
    if not st.session_state.api_key:
        raise ValueError("Please provide a valid Gemini API key.")

    response = st.session_state.model.predict(
        prompt=f"Generate a long, comprehensive, structured chapter for the following section:\n\n<section_title>{prompt}</section_title>",
        temperature=0.5,  # Adjust temperature for creativity
        max_output_tokens=8000,
    )

    return response.text

# Initialize
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

if 'button_text' not in st.session_state:
    st.session_state.button_text = "Generate"

if 'statistics_text' not in st.session_state:
    st.session_state.statistics_text = ""


st.write("""
# GeminiBook: Write full books using Gemini on Google AI
""")

def disable():
    st.session_state.button_disabled = True

def enable():
    st.session_state.button_disabled = False

def empty_st():
    st.empty()

try:
    if st.button('End Generation and Download Book'):
        if "book" in st.session_state:

            # Create markdown file
            markdown_file = create_markdown_file(st.session_state.book.get_markdown_content())
            st.download_button(
                label='Download Text',
                data=markdown_file,
                file_name='generated_book.txt',
                mime='text/plain'
            )

            # Create pdf file (styled)
            pdf_file = create_pdf_file(st.session_state.book.get_markdown_content())
            st.download_button(
                label='Download PDF',
                data=pdf_file,
                file_name='generated_book.pdf',
                mime='text/plain'
            )
        else:
            raise ValueError("Please generate content first before downloading the book.")


    with st.form("groqform"):
        if not GEMINI_API_KEY:
            gemini_input_key = st.text_input("Enter your Gemini API Key (gsk_yA...):", "",type="password")

        topic_text = st.text_input("What do you want the book to be about?", "")

        # Generate button
        submitted = st.form_submit_button(st.session_state.button_text,on_click=disable,disabled=st.session_state.button_disabled)
        
        # Statistics
        placeholder = st.empty()
        def display_statistics():
            with placeholder.container():
                if st.session_state.statistics_text:
                    if "Generating structure in background" not in st.session_state.statistics_text:
                        st.markdown(st.session_state.statistics_text+"\n\n---\n") # Format with line if showing statistics
                    else:
                        st.markdown(st.session_state.statistics_text)
                else:
                    placeholder.empty()

        if submitted:
            if len(topic_text)<10:
                raise ValueError("Book topic must be at least 10 characters long")

            st.session_state.button_disabled = True
            # st.write("Generating structure in background....")
            st.session_state.statistics_text = "Generating structure in background...." # Show temporary message before structure is generated and statistics show
            display_statistics()

            if not GEMINI_API_KEY:
                st.session_state.model = GenerativeModel(
                    model_name=MODEL_NAME,
                    generation_config=GenerationConfig(
                        temperature=1,  # Adjust these as needed
                        top_p=0.95,
                        top_k=64,
                        max_output_tokens=8192,
                        response_mime_type="text/plain",
                    ),
                    api_key=gemini_input_key
                )

            book_structure_json = generate_book_structure(topic_text)

            # st.session_state.statistics_text = str(large_model_generation_statistics)
            # display_statistics()

            total_generation_statistics = GenerationStatistics(model_name=MODEL_NAME)

            try:
                book = Book(book_structure_json)
                
                if 'book' not in st.session_state:
                    st.session_state.book = book

                # Print the book structure to the terminal to show structure
                print(json.dumps(book_structure_json, indent=2))

                st.session_state.book.display_structure()

                def stream_section_content(sections):
                    for title, content in sections.items():
                        if isinstance(content, str):
                            content_stream = generate_section(title+": "+content)
                            st.session_state.book.update_content(title, content_stream)
                        elif isinstance(content, dict):
                            stream_section_content(content)

                stream_section_content(book_structure_json)
            
            except json.JSONDecodeError:
                st.error("Failed to decode the book structure. Please try again.")

            enable()

except Exception as e:
    st.session_state.button_disabled = False
    st.error(e)

    if st.button("Clear"):
        st.rerun()
