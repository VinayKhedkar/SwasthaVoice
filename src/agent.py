import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import (
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-Maven-008976")

load_dotenv(".env.local")


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Persona and Style
You are an empathetic medical assistant coordinator. Your goal is to assess a patient's recovery through natural conversation and determine the next steps for their care. You speak naturally in a mix of Hindi and English, known as Hinglish, to make the patient feel comfortable.

Voice-First Formatting

Provide all responses in plain text only.

Never use markdown, bullet points, bolding, or special characters.

Spell out all numbers, such as saying twenty twenty-six instead of the digits.

Keep your responses short, usually one to three sentences, so the patient can easily follow the conversation.


Medical Conversational Flow

Start by asking the patient a broad question about how they are feeling today.

Conduct a gentle inquiry by asking about specific symptoms, energy levels, or their ability to perform daily tasks.

Analyze their responses internally. Do not ask the user if they need a follow-up. Instead, if you detect lingering symptoms or complications, suggest a follow-up or physical visit as a professional recommendation.

If you determine they are fully recovered based on their positive feedback, confirm this with them before closing the conversation.

Only ask one question at a time to keep the interaction simple and gather detailed information.

Multilingual Guidelines

Mirror the patient's language style.

If the patient uses Hindi terms for symptoms, acknowledge them in Hindi but keep the overall coordination professional.

Avoid medical jargon that might be hard for a text-to-speech system to pronounce clearly.

Tool Usage and Privacy

Execute tools like scheduling or marking recovery silently in the background only after you have reached a conclusion in the conversation.

When a tool confirms an action, summarize the result in a friendly way without using technical IDs or confirmation codes unless specifically asked.

Remind the patient that your guidance is general and they should seek immediate help for emergencies""",
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user with a Namaste! and tell the user that you are here to take a follow up on the user's health.",
            allow_interruptions=True,
        )
    

    @function_tool(name="get_summary")
    async def _http_tool_get_summary(self, context: RunContext) -> str:
        """
        Use this tool at the end of the call to draft a structured summary of the conversation.
        Call this tool when the conversation is concluding — either the patient has confirmed recovery,
        a follow-up has been recommended, or the patient wants to end the call.

        Returns a JSON string with the following structure:
        {
            "follow_up_needed": "needed | recovered | physical-visit | emergency",
            "conversation_summary": "Brief overview of the patient's health status",
            "conversation": [
                {"speaker": "agent" | "user", "text": "verbatim dialogue text"},
                ...
            ]
        }
        """

        from livekit.agents.llm import ChatContext as LKChatContext

        chat_messages = self.session.history.messages()

        conversation_lines = []
        for msg in chat_messages:
            role = msg.role

            text = msg.text_content
            if not text:
                text = " ".join(
                    part for part in msg.content
                    if isinstance(part, str)
                ).strip()

            if role and text:
                conversation_lines.append(f"{role.upper()}: {text}")

        conversation_text = (
            "\n".join(conversation_lines) if conversation_lines
            else "No conversation recorded."
        )
        logger.info("Extracted %d message turns for summary", len(conversation_lines))

        summary_prompt = f"""You are a medical records assistant. Analyze the following conversation between a medical coordinator (assistant) and a patient, then produce a JSON summary.

    CONVERSATION:
    {conversation_text}

    Instructions:
    1. Determine the appropriate follow_up_needed value:
    - "recovered": patient reports feeling well with no lingering symptoms
    - "needed": patient has mild/moderate symptoms that warrant monitoring
    - "physical-visit": patient has symptoms that need in-person examination
    - "emergency": patient describes severe/urgent symptoms

    2. Write a concise conversation_summary (2-4 sentences) covering:
    - Chief complaints or symptoms mentioned
    - Energy levels and daily functioning
    - Overall recovery status

    3. Reconstruct the conversation array with verbatim speaker turns.
    Use "agent" for the medical coordinator and "user" for the patient.

    Return ONLY valid JSON with no extra text, markdown, or code fences.

    Example format:
    {{
        "follow_up_needed": "needed",
        "conversation_summary": "Patient reports mild fatigue and occasional headaches three days post-discharge.",
        "conversation": [
            {{"speaker": "agent", "text": "Namaste! Aaj aap kaisa feel kar rahe hain?"}},
            {{"speaker": "user", "text": "Thoda thaka hua feel ho raha hai."}}
        ]
    }}"""

        try:
            summary_chat_ctx = LKChatContext.empty()
            summary_chat_ctx.add_message(role="user", content=summary_prompt)

            llm_client = inference.LLM(model="google/gemini-2.5-flash")
            chunks: list[str] = []
            async with llm_client.chat(chat_ctx=summary_chat_ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        chunks.append(chunk.delta.content)

            summary_json_str = "".join(chunks).strip()

            if summary_json_str.startswith("```"):
                lines = summary_json_str.splitlines()
                summary_json_str = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            summary_data = json.loads(summary_json_str)

        except json.JSONDecodeError as e:
            logger.error("LLM returned invalid JSON for summary: %s", e)
            summary_data = {
                "follow_up_needed": "needed",
                "conversation_summary": "Unable to parse structured summary. Manual review required.",
                "conversation": [],
                "parse_error": str(e),
            }
            summary_json_str = json.dumps(summary_data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Unexpected error generating summary: %s", e)
            summary_data = {
                "follow_up_needed": "needed",
                "conversation_summary": "Summary generation failed. Manual review required.",
                "conversation": [],
                "error": str(e),
            }
            summary_json_str = json.dumps(summary_data, ensure_ascii=False, indent=2)

        # Persist to disk
        try:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("data", f"{timestamp}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            logger.info("Conversation summary saved to %s", filepath)
        except OSError as e:
            logger.error("Failed to save summary to disk: %s", e)

        return summary_json_str
    
    


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="Maven-008976")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(
            model="openai/gpt-4.1-mini",
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en-US",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)