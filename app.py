import flet as ft
import json
from my_openai import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain ,SimpleSequentialChain
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS

embeddings = GooglePalmEmbeddings(google_api_key  = 'AIzaSyAUcyULNV5bARTN2KM9SLP3MWrz1jKJx3Q')
llm = OpenAI()

storage = FAISS.load_local('FAISS_DB',embeddings=embeddings)


def search(q):
    prompt1 = \
'''\
You are a legal translator, you should extract the legal question from the text in json format and add some details related to the legal question so that we can put the result in the search engine.

for example :
    text : 'ما هو حكم او عقوبة من ينظم عصابة لقتل الناس ؟'
    AI : {
        q : What is the ruling or punishment for someone who organizes a gang to kill people? He organizes a group to kill a group of people in public places or their homes! He kills people!
    }
    Give me result as json only.
'''
    prompt2 =\
'''
text : {q_q}
AI:\
'''
    if q == None:
        q = 'None?'
    prompt = prompt1 + prompt2.format(q_q=q)
    result = llm(prompt)
    result = json.loads(result)['q']    
    final_result =storage.similarity_search(result,k=7)
    final_result2 = ''
    for i in final_result:
        final_result2+= i.page_content
        final_result2+='\n\n'
    return final_result2

prompt = \
'''\
You are a legal specialist, your name is Mona
You must answer the question according to the context
context : {context}

Question : {question}
Useful Full Answer:
'''

prompt_template = PromptTemplate(input_variables=['question' ,'context'], template=prompt ,)

answer_chain = LLMChain(llm=llm , prompt=prompt_template , )

prompt = \
'''\
You are an expert translator. You must translate the answer into the same language as the question
You must translate the answer into Arabic

answer : {answer}
question: {question}
give answer after translation only!
'''

prompt_template = PromptTemplate(input_variables=['answer' ,'question'], template=prompt ,)

translate_chain = LLMChain(llm=llm , prompt=prompt_template , )

def answer(q):
    docs = search(q)
    answer = answer_chain(dict(question = q , context = docs))['text']
    
    translat_answer = translate_chain( inputs = dict(answer = answer , question = q ,))['text']
    return translat_answer






class Chat(ft.UserControl):
    def build(self):
        self.heading = ft.Text(value="Chat to Legal Specialist", size=24)
        self.text_input = ft.TextField(hint_text="Enter your qustion", expand=True, multiline=True)
        self.output_column = ft.Column()
        self.scroll = True
        return ft.Column(
            width=800,
            controls=[
                self.heading,
                ft.Row(
                    controls=[
                        self.text_input,
                        ft.ElevatedButton("Submit", height=60, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=1)), on_click=self.btn_clicked),
                    ],
                ),
                self.output_column,
            ],
        )
    
    def btn_clicked(self, event):
        completion = answer(self.text_input.value)
        self.output = completion
        
        result = Output(self.output, self.text_input.value, self.outputDelete)
        
        self.output_column.controls.append(result)
        self.text_input.value = ""
        self.update()

    def outputDelete(self, result):
        self.output_column.controls.remove(result)
        self.update()


class Output(ft.UserControl):
    def __init__(self, myoutput, mytext_input, myoutput_delete):
        super().__init__()
        self.myoutput = myoutput 
        self.mytext_input = mytext_input
        self.myoutput_delete = myoutput_delete

    def build(self):
        self.output_display = ft.Text(value=self.myoutput, selectable=True)
        self.delete_button = ft.IconButton(ft.icons.DELETE_OUTLINE_SHARP, on_click=self.delete)
        self.input_display = ft.Container(ft.Text(value=self.mytext_input), bgcolor=ft.colors.GREY_900, padding=10)
        self.display_view = ft.Column(controls=[self.input_display, self.output_display, self.delete_button])
        return self.display_view

    def delete(self, e):
        self.myoutput_delete(self)


def main(page):
    page.scroll = True
#     page.window_width = 700
#     page.window_height = 900
    mychat = Chat()
    page.add(mychat)

ft.app(target=main)