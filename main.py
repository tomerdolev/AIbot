# יבוא כלים מספריית langchain 
# Embeddings הגדרתי פה את כל הכלים שנזדקק להם- טעינת קבצים, חלוקת טקסטים, אחסון ויצירת
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

# טעינת משתני סביבה מקובץ .env
load_dotenv()

# נכניס לרשימה את כל המסמכים שנטען
# נסרוק את תיקיית הדאטה, כל המסמכים שנמצאו נשמרים ברשימה גדולה של טקסטים
# יצרתי הפרדה בין קבצי docs\PDF
all_docs = []
data_folder = "data"

#לולאה על כל הקבצים בתיקיית דאטה
for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_docs.extend(documents)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        all_docs.extend(documents)


# מחלקת את המקטעים לגודל אחיד וחפיפה מוגדרת- overlap
# יאפשר לנו ליצור embaddings בצורה אופטימלית
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

# יצירת Embadding 
# כל מקטע מומר לוקטור מספרי שמתאר את המשמעות שלו בשפה טבעית
# השתמשתי פה במפתח API של OPENAI
api_key = os.getenv("OPEN_AI_SECRET_KEY")
if not api_key:
    raise ValueError("OPEN_AI_SECRET_KEY environment variable is not set")
    
embedding = OpenAIEmbeddings(openai_api_key=api_key)
# נשמור את הוקטורים במאגר FAISS
# תשמש עבורנו כמנוע חיפוש וקטורי שיאפשר שליפה מהירה של טקסטים דומים
db = FAISS.from_documents(docs, embedding)
db.save_local("vector_store")

#הודעה זו נועדה עבורי רק לשם אימות שאראה שהכל תקין בפיתוח המודל עד כאן
print("הקבצים נותחו ונשמרו.")

# ביצוע שאילתא על המאגר הוקטורי
query = "הסבר לי מה זה משוואת לינארית?"

# similarity_search מבצע מאחורי הקלעים המרה של השאילתא לembaddings 
# תבוצע השוואת דמיון בין הוקטורים הקיימים במאגר לבין השאילתא לאחר המרתה בembaddings
# מדרג את התוצאות לפי קרבה סמנטית ומחזיר לנו את הקטעים הכי דומים

docs = db.similarity_search(query, k=3)  # מחפש 3 תוצאות הכי קרובות

#לולאה שתדפיס 3 תשובות הכי רלוונטיות
for i, doc in enumerate(docs):
    print(f"--- תוצאה {i+1} ---")
    print(doc.page_content)
    print()
