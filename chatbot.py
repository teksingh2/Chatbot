from langchain.agents import Tool, AgentExecutor, ConversationalAgent
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta
import re
import os
from typing import Dict, Optional
import phonenumbers

class UserDataCollector:
    def __init__(self):
        self.required_fields = ['name', 'phone', 'email']
        self.user_data = {}
        self.current_field = None
        
    def validate_email(self, email: str) -> bool:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def validate_phone(self, phone: str) -> bool:
        try:
            parsed_number = phonenumbers.parse(phone, "US")
            return phonenumbers.is_valid_number(parsed_number)
        except:
            return False
            
    def validate_name(self, name: str) -> bool:
        return bool(name.strip() and len(name) >= 2)
    
    def get_next_required_field(self) -> Optional[str]:
        for field in self.required_fields:
            if field not in self.user_data:
                return field
        return None
    
    def process_input(self, user_input: str) -> Dict:
        if not self.current_field:
            self.current_field = self.get_next_required_field()
            
        if self.current_field:
            is_valid = False
            if self.current_field == 'email':
                is_valid = self.validate_email(user_input)
            elif self.current_field == 'phone':
                is_valid = self.validate_phone(user_input)
            elif self.current_field == 'name':
                is_valid = self.validate_name(user_input)
                
            if is_valid:
                self.user_data[self.current_field] = user_input
                self.current_field = self.get_next_required_field()
                
            return {
                'is_valid': is_valid,
                'current_field': self.current_field,
                'completed': self.current_field is None,
                'user_data': self.user_data
            }

class DateExtractor:
    def __init__(self):
        self.day_mapping = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }
    
    def parse_relative_date(self, text: str) -> Optional[str]:
        text = text.lower()
        today = datetime.now()
        
        if 'next' in text:
            for day, offset in self.day_mapping.items():
                if day in text:
                    days_ahead = offset - today.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target_date = today + timedelta(days=days_ahead)
                    return target_date.strftime('%Y-%m-%d')
                    
        return None

class Chatbot:
    def __init__(self, documents_dir: str):
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                                        google_api_key="AIzaSyDeF7Oo27-qNRxM4ofg-_nFXOXmrYXfYiQ")
        
        # Initialize document processing
        self.initialize_document_qa(documents_dir)
        
        # Initialize components
        self.user_collector = UserDataCollector()
        self.date_extractor = DateExtractor()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize tools
        self.tools = [
            Tool(
                name="Document QA",
                func=self.query_documents,
                description="Useful for answering questions about documents"
            ),
            Tool(
                name="Date Extraction",
                func=self.extract_date,
                description="Useful for extracting dates from text"
            ),
            Tool(
                name="Book Appointment",
                func=self.book_appointment,
                description="Useful for booking appointments"
            )
        ]
        
        # Initialize agent
        self.agent = ConversationalAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory
        )
        
    def initialize_document_qa(self, documents_dir: str):
        # Load and process documents
        loader = DirectoryLoader(documents_dir)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.vectorstore = FAISS.from_documents(texts, embeddings)
        
    def query_documents(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query)
        return docs[0].page_content if docs else "No relevant information found."
       
    def extract_date(self, text: str) -> str:
        date = self.date_extractor.parse_relative_date(text)
        return date if date else "Could not extract date from the text."
        
    def handle_call_request(self, user_input: str) -> str:
        result = self.user_collector.process_input(user_input)
        
        if not result['is_valid'] and result['current_field']:
            field_prompts = {
                'name': "Please enter your name:",
                'phone': "Please enter your phone number:",
                'email': "Please enter your email address:"
            }
            return field_prompts[result['current_field']]
            
        if result['completed']:
            return (f"Thank you! We'll contact you shortly at:\n"
                   f"Name: {result['user_data']['name']}\n"
                   f"Phone: {result['user_data']['phone']}\n"
                   f"Email: {result['user_data']['email']}")
                   
        return "Please provide valid information."
        
    def process_message(self, message: str) -> str:
        # Check if we're in the middle of collecting user information
        if self.user_collector.current_field:
            return self.handle_call_request(message)
            
        # Check if the message is a call request
        if "call" in message.lower():
            self.user_collector.current_field = 'name'
            return "I'll help you schedule a call. Please enter your name:"
            
        # Otherwise, process with the agent
        try:
            response = self.agent_executor.run(message)
            return response
        except Exception as e:
            return f"I encountered an error: {str(e)}"
    def book_appointment(self, text: str) -> str:
        return "I can help you book an appointment. Please provide the date and time."
# Example usage
def main():
    # Initializing the chatbot with a documents directory
    chatbot = Chatbot("/Users/teksinghayer/Desktop/palm_tech_chatbot/pdf")
    chatbot.initialize_document_qa("/Users/teksinghayer/Desktop/palm_tech_chatbot/pdf")
    print("Chatbot initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.process_message(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
    