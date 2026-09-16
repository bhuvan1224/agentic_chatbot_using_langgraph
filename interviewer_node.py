from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages

# ==========================================
# 1. DEFINE THE UPDATED DATA MODEL
# ==========================================
class ResumeInfo(BaseModel):
    full_name: Optional[str] = Field(default=None, description="The user's full name.")
    contact_info: Optional[str] = Field(default=None, description="Phone number, email, city, LinkedIn, GitHub.")
    target_role: Optional[str] = Field(default=None, description="Target job title (e.g., Data Scientist, AI Engineer).")
    job_description: Optional[str] = Field(default=None, description="The specific Job Description (JD) text or key requirements.")
    education: Optional[str] = Field(default=None, description="Degrees, university names, CGPA/marks, and graduation years.")
    work_experience: Optional[str] = Field(default=None, description="Previous internships, roles, company names, dates, and duties.")
    projects: Optional[str] = Field(default=None, description="Key projects built, technologies used, links, and descriptions.")
    skills: Optional[List[str]] = Field(default=None, description="Technical skills, programming languages, frameworks, and tools.")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    resume_data: ResumeInfo
    tailored_resume: str

# ==========================================
# 2. THE ULTIMATE HALLUCINATION FILTER
# ==========================================
def is_valid_extracted_data(value):
    """Checks if the LLM hallucinated placeholder text or 'N/A'."""
    if not value:
        return False
        
    val_str = str(value).lower().strip()
    
    # 💡 Added all variations of "Not provided" or "Unknown"
    bad_phrases = [
        "your full name", "string", "none", "null", "the job title", 
        "language1", "degree", "university name", "company name", 
        "your phone number", "email address", "n/a", "not provided", 
        "unknown", "not specified", "not mentioned", "na", "blank"
    ]
    
    if isinstance(value, list):
        if len(value) == 0: 
            return False
        if str(value[0]).lower().strip() in bad_phrases: 
            return False
            
    for phrase in bad_phrases:
        if phrase in val_str:
            return False
            
    # Block single-letter hallucinations
    if isinstance(value, str) and len(val_str) < 2:
        return False
        
    return True

# ==========================================
# 3. INTERVIEWER NODE LOGIC
# ==========================================
def get_interviewer_node(primary_llm):
    llm_with_extraction = primary_llm.with_structured_output(ResumeInfo)
    
    def interviewer_node(state: AgentState):
        messages = state.get("messages", [])
        current_resume = state.get("resume_data", ResumeInfo())

        # --- STEP 1: EXTRACT DATA ---
        if messages:
            try:
                user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
                if user_msgs:
                    # 💡 STRICTER PROMPT: Forcing the LLM to output null for greetings/general requests
                    extraction_prompt = SystemMessage(
                        content="You are a strict data extractor. Extract details ONLY from the user's messages. "
                                "CRITICAL INSTRUCTION: If a specific piece of information is missing, you MUST output literal `null`. "
                                "DO NOT output 'N/A', 'Not provided', 'Unknown', or invent any placeholders. "
                                "If the user is just saying a greeting (like 'hello') or making a general request (like 'I want a resume'), output `null` for ALL fields."
                    )
                    
                    updated_data = llm_with_extraction.invoke([extraction_prompt] + user_msgs)
                    
                    for field, value in updated_data.model_dump().items():
                        if is_valid_extracted_data(value): 
                            setattr(current_resume, field, value)
            except Exception as e:
                print(f"Extraction error: {e}") 

        # --- STEP 2: CHECK FOR MISSING FIELDS ---
        missing_fields = []
        if not current_resume.target_role and not current_resume.job_description:
            missing_fields.append("Target Role or Job Description")
            
        for field in ["full_name", "contact_info", "education", "work_experience", "projects", "skills"]:
            if not getattr(current_resume, field):
                missing_fields.append(field.replace("_", " "))

        # --- STEP 3: EXIT CONDITION ---
        if not missing_fields:
            return {
                "messages": [AIMessage(content="I have extracted all your details and projects! Generating your 1-page ATS-optimized resume now...")],
                "resume_data": current_resume
            }

        # --- STEP 4: ASK FOR MISSING DATA ---
        system_prompt = f"""You are an expert ATS Career Coach.
        You are gathering missing information to build a tailored 1-page ATS resume.
        
        Missing information: {', '.join(missing_fields)}.
        
        RULES:
        1. If 'Target Role or Job Description' is missing, specifically ask the user: "What specific job title or Job Description (JD) are you applying for so I can tailor your ATS keywords?"
        2. Ask for ONLY ONE OR TWO missing items at a time in a polite, professional manner.
        """
        
        chat_response = primary_llm.invoke([SystemMessage(content=system_prompt)] + messages)
        
        return {
            "messages": [chat_response],
            "resume_data": current_resume
        }
        
    return interviewer_node