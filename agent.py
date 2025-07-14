from os import getenv
import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_core.documents import Document

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END

from CalendarClient import CalendarClient

load_dotenv()
TOKEN = getenv("OPENAI_API_KEY")

calendar = CalendarClient(
    service_account_file="service-creds.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    calendar_id=""
)

@tool
def list_free_slots(start_iso: str, end_iso: str, duration_minutes: int) -> str:
    """
    Повертає список вільних слотів у календарі між двома ISO-датами з вказаною тривалістю.

    Parameters:
    ----------
    start_iso : str
        Початкова дата/час у форматі ISO 8601 (наприклад, "2025-06-16T09:00:00+03:00").
    
    end_iso : str
        Кінцева дата/час у форматі ISO 8601, яка визначає діапазон для пошуку вільних слотів.
    
    duration_minutes : int
        Тривалість бажаного слоту у хвилинах (наприклад, 30, 60).

    Returns:
    -------
    str
        Список доступних часових слотів або повідомлення, що вільного часу немає.
    """
    return calendar.list_free_slots(
        start_iso=start_iso,
        end_iso=end_iso,
        duration_minutes=duration_minutes
    )

@tool
def create_appointment(specialist: str, start_iso: str, end_iso: str, summary: str, description: str, attendee_email: str = None,) -> str:
    """
    Створює подію в Google Calendar на вказану дату з описом і деталями.

    Parameters:
    ----------
    specialist : str
        Ім'я або ідентифікатор спеціаліста, який проводить процедуру (наприклад, "laserepilation").
        Це поле може використовуватися для логіки маршрутизації або запису до відповідного календаря.

    start_iso : str
        Початкова дата/час події у форматі ISO 8601 
        (наприклад, "2025-06-16T10:00:00+03:00").

    end_iso : str
        Кінцева дата/час події у форматі ISO 8601.

    summary : str
        Короткий опис події або її назва 
        (наприклад, "Лазерна епіляція для Олени").

    description : str
        Детальний опис події, включаючи тип процедури, побажання клієнта, 
        контактну інформацію тощо.

    attendee_email : str, optional
        Email клієнта, якому буде надіслано запрошення на подію. 
        Якщо не вказано, подія створюється без учасника.

    Returns:
    -------
    str
        Підтвердження про успішне створення події або повідомлення про помилку.
    """
    return calendar.create_appointment(
        specialist=specialist,
        start_iso=start_iso,
        end_iso=end_iso,
        summary=summary,
        description=description,
        attendee_email=attendee_email
    )

@tool
def current_date_info():
    """Отримує поточну дату та назву дня тижня. Використовуй цей інструмент, коли потрібно знати, який сьогодні день або дата."""
    current_date = datetime.date.today()
    day_name = current_date.strftime("%A")
    return {day_name: str(current_date)}

faq_text = open("faq.txt", "r", encoding="utf-8").read()
docs = [Document(page_content=chunk) for chunk in faq_text.split("\n\n") if chunk.strip()]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore.from_documents(chunks, embedding_model)

retriever = VectorStoreRetriever(vectorstore=vectorstore)

api_tools = [current_date_info, list_free_slots, create_appointment]
tool_mapping = {tool.name: tool for tool in api_tools}

# llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(api_tools)
llm = ChatOpenAI(model="o3-mini").bind_tools(api_tools)

checkpointer = InMemorySaver()

class State(TypedDict, total=False):
    user_input: str
    output: str
    messages: list

def beauty_agent(state: State) -> State:
    print("🟢 STATE IN:", state)

    if "messages" not in state:
        state["messages"] = [
            SystemMessage(content="""Ти - менеджер майстра лазерної епіляції. Спілкуєшся українською, чітко, лаконічно та доброзичливо (додавай смайлики). 
            Твоя роль: 
            1. Вести діалог з клієнтами в месенджерах від імені майстра;
            2. Пропонувати процедури, відповідати на запитання, відпрацьовувати заперечення;
            3. Якщо клієнт погоджується — бронюєш слот у Google Calendar. Якщо час зайнятий - запропонуй інши доступні варіанти. Перед бронюванням обов'язково перевір поточну дату інструментом "current_date_info".
            Після успішного бронювання слоту, надсилай повідомлення:«Запис підтверджено ✅ Чекаємо вас DD.MM о HH\:MM»
            Робочий графік з 9:00 до 18:00.
            Відповідай коротко. Використовуй емодзі, пиши тепло й професійно.
            """)
        ]

    query = state["user_input"]
    retrieved_docs = retriever.get_relevant_documents(state["user_input"])
    if retrieved_docs:
        rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        state["messages"].append(
            AIMessage(content=f"Контекст з FAQ:\n{rag_context}")
        )

    state["messages"].append(HumanMessage(content=state["user_input"]))

    # Примусовий виклик current_date_info()
    # date_info = current_date_info.invoke(state["messages"])
    # state["messages"].append(ToolMessage(content={"result": date_info}, tool_call_id="manual-date-check"))
    while True:
        llm_output = llm.invoke(state["messages"])
        state["messages"].append(llm_output)

        if llm_output.tool_calls:
            for call in llm_output.tool_calls:
                tool_fn = tool_mapping.get(call["name"])
                if tool_fn:
                    tool_output = tool_fn.invoke(call["args"])
                    state["messages"].append(
                        ToolMessage(content={"result": tool_output}, tool_call_id=call["id"])
                    )
        else:
            state["output"] = llm_output.content
            break

    return state

builder = StateGraph(state_schema=State)
builder.add_node("BeautyAgent", beauty_agent)
builder.set_entry_point("BeautyAgent")
builder.add_edge("BeautyAgent", END)

graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    result = graph.invoke({"user_input": "Чи є у вас вільні години на 14 липня? В мене є година часу."}, {"configurable": {"thread_id": "1"}})
    result = graph.invoke({"user_input": "Я б хотів записатись на 14 липня на 13:00"}, {"configurable": {"thread_id": "1"}})
    print("🟢 RESULT:", result["output"])
