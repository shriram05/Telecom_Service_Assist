import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from utils.document_loader import DocumentLoader
from dotenv import load_dotenv
import tempfile
from datetime import datetime
import sqlite3
load_dotenv()
# Add parent directory to path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))
 
# Import core functionality
from orchestration.graph import create_graph, TelecomAssistantState

# Set page configuration
st.set_page_config(
    page_title="Telecom Service Assistant",
    page_icon="📱",
    layout="wide"
)
conn = sqlite3.connect('telecom.db')  # Use the actual database path
cursor = conn.cursor()


# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "email" not in st.session_state:
    st.session_state.email = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph" not in st.session_state:
    # Initialize the LangGraph workflow
    st.session_state.graph = create_graph()
 
# Function to process user queries
def process_query(query: str):
    """Process a user query through the LangGraph workflow"""
    # Create initial state
    state = {
        "query": query,
        "customer_info": {"email": st.session_state.email},
        "classification": "",
        "intermediate_responses": {},
        "final_response": "",
        "chat_history": st.session_state.chat_history
    }
   
    # Process through the graph
    # In a real implementation, you'd use:
    # result = st.session_state.graph.invoke(state)
    # return result["final_response"]
    try:
        # Process through the graph - this time we'll actually use it
        result = st.session_state.graph.invoke(state)
        
        # The result should contain a final_response
        if result and "final_response" in result:
            return result["final_response"]
        else:
            # If something went wrong, fall back to the classification-based responses
            return fallback_response(query)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return fallback_response(query)


def fallback_response(query: str):
    """Generate a fallback response based on query content"""
    query_lower = query.lower()
   
    if any(word in query_lower for word in ["bill", "charge", "payment", "account"]):
        # Instead of hardcoded response, call the billing agent directly
        
        # customer_email = "CUST001"  # For demo purposes
        return ""
    elif any(word in query_lower for word in ["network", "signal", "connection", "call", "data", "slow"]):
        return "I've checked our network status, and there's scheduled maintenance in your area that might be affecting your connectivity. This should be resolved by 6 PM today. In the meantime, try switching to 3G mode in your phone settings for more stable connectivity."
    elif any(word in query_lower for word in ["plan", "recommend", "best", "upgrade", "family"]):
        return "For a family of four with heavy streaming usage, I recommend our Family Share Plus plan at $89.99/month. This includes 40GB of shared high-speed data, unlimited talk and text for all lines, and free access to our premium streaming service."
    elif any(word in query_lower for word in ["how", "what", "configure", "setup", "apn", "volte"]):
        return "To enable VoLTE on your Android device, go to Settings > Connections > Mobile Networks > VoLTE calls and toggle it on. Make sure your device is VoLTE compatible and you're in a coverage area. This will improve call quality and allow simultaneous voice and data usage."
    else:
        return "I'm not sure how to help with that specific question. Could you try rephrasing or ask about our services, billing, network issues, or technical support?"
    
# Sidebar for authentication
with st.sidebar:
    st.title("Telecom Service Assistant")
   
    if not st.session_state.authenticated:
        st.subheader("Login")
        email = st.text_input("Email Address")
        user_type = st.selectbox("User Type", ["Customer", "Admin"])
       
        if st.button("Login"):
            if email and "@" in email:
                st.session_state.authenticated = True
                st.session_state.user_type = user_type
                st.session_state.email = email
                st.success(f"Logged in as {user_type}")
                st.rerun()
            else:
                st.error("Please enter a valid email address")
    else:
        st.success(f"Logged in as {st.session_state.user_type}")
        st.text(f"Email: {st.session_state.email}")
       
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.email = None
            st.session_state.chat_history = []
            st.rerun()
 
# Main app content
if st.session_state.authenticated:
    if st.session_state.user_type == "Customer":
        st.title("Welcome to Telecom Service Assistant")
       
        tab1, tab2, tab3 = st.tabs(["Chat Assistant", "My Account", "Network Status"])
       
        with tab1:
            st.header("Chat with our AI Assistant")
           
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
           
            # Chat input
            if prompt := st.chat_input("How can I help you today?"):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
               
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
               
                # Process user query through LangGraph
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = process_query(prompt)
                        st.write(response)
               
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
       
        with tab2:
            customer_mail = st.session_state.email
            cursor.execute("""Select customer_id, service_plan_id, last_billing_date from customers where email = ?""",(customer_mail,))
            data = cursor.fetchone()
            if data:
                customer_id, service_plan_id, last_billing_date = data
                # Proceed with further processing
                st.write(f"Customer ID: {customer_id}")
               
            cursor.execute("""Select data_used_gb, voice_minutes_used, sms_count_used, total_bill_amount from customer_usage where customer_id = ?""", (customer_id,))
            ans = cursor.fetchone()
            data_used_gb, voice_minutes_used, sms_count_used, total_bill_amount = ans 

            st.header("My Account Information")
            st.subheader("Current Plan")
            st.write({service_plan_id})
           
            col1, col2, col3 = st.columns(3)
            with col1:
                rem_gb = 20 - data_used_gb
                st.metric("Data Used", f"{data_used_gb} GB", f"{rem_gb} GB remaining")
            with col2:
                st.metric("Voice Minutes", f"{voice_minutes_used} mins", "Unlimited")
            with col3:
                st.metric("SMS Used", f"{sms_count_used}", "Unlimited")
               
            st.subheader("Billing Information")
            st.write("Next Bill Date:", f"{last_billing_date}")
            st.write("Monthly Charge: ₹ ", f"{total_bill_amount}",".00")
       
        with tab3:
            st.header("Network Status")
            status_df = pd.DataFrame({
                "Region": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
                "4G Status": ["Normal", "Normal", "Degraded", "Normal", "Normal"],
                "5G Status": ["Normal", "Maintenance", "Normal", "Normal", "Degraded"]
            })
           
            st.dataframe(status_df, use_container_width=True)
           
            st.subheader("Known Issues")
            st.info("Scheduled maintenance in Delhi region (03:00-05:00 AM)")
            st.warning("Network congestion reported in Bangalore South")
           
    elif st.session_state.user_type == "Admin":
        st.title("Admin Dashboard")
       
        tab1, tab2, tab3 = st.tabs(["Knowledge Base Management", "Customer Support", "Network Monitoring"])
       
        with tab1:
            st.header("Knowledge Base Management")
           
            st.subheader("Upload Documents to Knowledge Base")
            uploaded_file = st.file_uploader("Upload PDF, Markdown, or Text files",
                                            type=["pdf", "md", "txt"],
                                            accept_multiple_files=True)
           
            if uploaded_file:
                for file in uploaded_file:
                    path = os.path.join("C:/Users/shriramkumar.an/Desktop/Telecom Service Assistant/telecomAssist/src/telecomassist/data/temp_dir", file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    with open(f"data/documents/{file.name}", "wb") as w:
                        w.write(file.getbuffer())
                    docLoader = DocumentLoader()
                    docLoader.process_uploads(path)
                    st.success(f"Processed {file.name} and added to knowledge base")
                    file_path = f"C:/Users/shriramkumar.an/Desktop/Telecom Service Assistant/telecomAssist/src/telecomassist/data/temp_dir/{file.name}"
                    os.remove(file_path)
            

           
            st.subheader("Existing Documents")
            data_folder_path = "C:/Users/shriramkumar.an/Desktop/Telecom Service Assistant/telecomAssist/src/telecomassist/data/documents"  # Adjust path as needed

# List to hold document details
            document_details = []

            # Loop through all files in the folder
            for file_name in os.listdir(data_folder_path):
                file_path = os.path.join(data_folder_path, file_name)
                
                # Check if it's a file and not a sub-folder
                if os.path.isfile(file_path):
                    # Get file details
                    file_extension = file_name.split('.')[-1]
                    last_updated = os.path.getmtime(file_path)  # Get the last modified time in seconds
                    last_updated_str = datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d')  # Convert to readable format

                    # Append the details to the list
                    document_details.append({
                        "Document Name": file_name,
                        "Type": file_extension.capitalize(),
                        "Last Updated": last_updated_str
                    })

            # Create the DataFrame
            doc_df = pd.DataFrame(document_details)

            st.dataframe(doc_df, use_container_width=True)
       
        with tab2:
            st.header("Customer Support Dashboard")
           
            st.subheader("Active Support Tickets")
            ticket_df = pd.DataFrame({
                "Ticket ID": ["TKT004", "TKT005"],
                "Customer": ["Ananya Singh", "Vikram Reddy"],
                "Issue": ["Account reactivation", "Slow internet speeds"],
                "Status": ["In Progress", "Assigned"],
                "Priority": ["Medium", "Medium"],
                "Created": ["2023-06-15", "2023-06-17"]
            })
           
            st.dataframe(ticket_df, use_container_width=True)
           
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Tickets", "2", "-3")
            with col2:
                st.metric("Avg. Resolution Time", "4.3 hours", "-0.5")
            with col3:
                st.metric("Customer Satisfaction", "92%", "+3%")
       
        with tab3:
            st.header("Network Monitoring")
           
            st.subheader("Active Network Incidents")
            incident_df = pd.DataFrame({
                "Incident ID": ["INC003"],
                "Type": ["Equipment Failure"],
                "Location": ["Delhi West"],
                "Affected Services": ["Voice, Data, SMS"],
                "Started": ["2023-06-15 08:15:00"],
                "Status": ["In Progress"],
                "Severity": ["Critical"]
            })
           
            st.dataframe(incident_df, use_container_width=True)
 
# Function to run the app
def main():
    # Already set up above
    pass
 
if __name__ == "__main__":
    main()