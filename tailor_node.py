from langchain_core.messages import AIMessage, SystemMessage
from interviewer_node import AgentState

def get_tailor_node(primary_llm):
    """
    Takes your LLM and returns the configured Tailor Node.
    """
    def tailor_node(state: AgentState):
        resume_data = state.get("resume_data")
        
        system_prompt = f"""You are an Executive ATS Resume Writer.
        Your job is to generate a HIGHLY COMPACT, ATS-FRIENDLY, EXACT 1-PAGE Markdown resume.

        TARGET ROLE / JD: {resume_data.target_role or 'Not specified'}
        JOB DESCRIPTION CONTEXT: {resume_data.job_description or 'Align keywords for the target role.'}

        RAW USER DATA:
        - Full Name: {resume_data.full_name}
        - Contact: {resume_data.contact_info}
        - Education: {resume_data.education}
        - Work Experience: {resume_data.work_experience}
        - Projects: {resume_data.projects}
        - Skills: {', '.join(resume_data.skills) if isinstance(resume_data.skills, list) else resume_data.skills}

        CRITICAL ATS & FORMATTING RULES (YOU MUST OBEY THESE STRICT LIMITS):
        1. STRICT 1-PAGE LIMIT: You MUST keep the entire response under 400 words. 
        2. MAX BULLET POINTS: Use MAXIMUM 2 bullet points per Project, and MAXIMUM 2 bullet points per Work Experience. 
        3. NO FLUFF: Remove filler words. Start every bullet with a strong action verb. Focus only on hard skills (Python, LangGraph, NLP, etc.) and quantifiable results.
        4. CLEAN STRUCTURE: Use standard markdown headers EXACTLY like this:
           # [Full Name]
           [Contact Info: Phone | Email | Location | LinkedIn | GitHub]
           
           ## Key Skills
           ## Professional Experience
           ## Projects
           ## Education
        5. OUTPUT ONLY THE MARKDOWN RESUME. Do not include introductory or concluding remarks.
        """
        
        response = primary_llm.invoke([SystemMessage(content=system_prompt)])
        resume_content = response.content
        
        final_message = f"**Your 1-Page ATS Resume is Ready!** 🎉\n\nYou can review it below and download the 1-Page PDF:\n\n---\n{resume_content}\n---"
        
        return {
            "messages": [AIMessage(content=final_message)],
            "tailored_resume": resume_content
        }

    return tailor_node