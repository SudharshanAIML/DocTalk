"""
Semantic Reasoner Module for Deep Search.

Performs multi-hop reasoning using Claude (Anthropic) as the backbone LLM.
Implements chain-of-thought prompting, relationship extraction,
and implicit information discovery.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# Try to import Anthropic - will be optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not installed. Install with: pip install anthropic")

# Fallback to Groq if Anthropic not available
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class ReasoningDepth(Enum):
    """Depth of reasoning to perform."""
    QUICK = "quick"       # Fast, minimal reasoning
    STANDARD = "standard" # Balanced reasoning
    DEEP = "deep"         # Thorough, multi-step reasoning
    EXHAUSTIVE = "exhaustive"  # Maximum reasoning depth


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    question: str
    reasoning: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Complete reasoning chain for a query."""
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    total_confidence: float
    reasoning_depth: ReasoningDepth
    processing_time_ms: int
    token_usage: Dict[str, int]
    metadata: Dict = field(default_factory=dict)


@dataclass
class Relationship:
    """A relationship between concepts or documents."""
    source: str
    target: str
    relationship_type: str  # causes, relates-to, contradicts, supports, depends-on
    strength: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)


@dataclass
class ImplicitInfo:
    """Implicit information derived through reasoning."""
    statement: str
    derivation: str  # How it was derived
    supporting_facts: List[str]
    confidence: float
    source_documents: List[str]


class SemanticReasoner:
    """
    Performs semantic reasoning using Claude or fallback LLMs.
    """
    
    # System prompts for different reasoning tasks
    CHAIN_OF_THOUGHT_PROMPT = """You are an expert reasoning assistant. Your task is to analyze documents and answer questions using careful, step-by-step reasoning.

IMPORTANT RULES:
1. Think through each step explicitly before answering
2. Cite specific evidence from the provided context
3. Acknowledge uncertainty when information is incomplete
4. Identify implicit information that can be derived from the context
5. Consider multiple perspectives and potential contradictions

FORMAT your reasoning as follows:
- **Step 1**: [First observation or analysis]
- **Step 2**: [Build on previous step]
- **Step N**: [Final synthesis]
- **Answer**: [Clear, comprehensive answer]

Be thorough but concise. Focus on what matters for answering the question."""

    RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert at identifying relationships between concepts, entities, and documents.

Analyze the provided context and identify ALL relationships. For each relationship, specify:
1. Source entity/concept
2. Target entity/concept
3. Relationship type (one of: causes, relates-to, contradicts, supports, depends-on, similar-to, part-of, leads-to)
4. Strength (0.0 to 1.0)
5. Evidence (quote from text)

Output as JSON array:
[{"source": "...", "target": "...", "type": "...", "strength": 0.X, "evidence": "..."}]"""

    IMPLICIT_INFO_PROMPT = """You are an expert at deriving implicit information from explicit statements.

Given the context, identify information that is NOT explicitly stated but can be logically derived.

For each implicit finding:
1. State the derived information
2. Explain how it was derived
3. List the supporting facts
4. Estimate confidence (0.0 to 1.0)

Focus on:
- Logical implications
- Unstated assumptions
- Consequences of stated facts
- Patterns across multiple statements

Output as JSON array:
[{"statement": "...", "derivation": "...", "supporting_facts": ["..."], "confidence": 0.X}]"""

    MULTI_HOP_PROMPT = """You are solving a multi-step reasoning problem. Answer each sub-question in sequence, using previous answers to inform later ones.

Context from documents:
{context}

Previous answers (if any):
{previous_answers}

Current sub-question: {question}

Think step by step:
1. What information from the context is relevant?
2. How do previous answers inform this question?
3. What is the logical answer?

Provide your reasoning, then a clear answer."""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        fallback_model: str = os.getenv("GROQ_MODEL", "groq/compound"),
        max_tokens: int = 4096,
        temperature: float = 0.3
    ):
        """
        Initialize the semantic reasoner.
        
        Args:
            anthropic_api_key: Anthropic API key (uses env var if not provided)
            groq_api_key: Groq API key for fallback
            model: Primary Claude model to use
            fallback_model: Fallback model for Groq
            max_tokens: Maximum tokens for responses
            temperature: Temperature for generation
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize clients
        self.anthropic_client = None
        self.groq_client = None
        
        if ANTHROPIC_AVAILABLE and self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            logger.info("Initialized Anthropic client")
        
        if GROQ_AVAILABLE and self.groq_api_key:
            self.groq_client = ChatGroq(
                model=self.fallback_model,
                temperature=self.temperature,
                groq_api_key=self.groq_api_key
            )
            logger.info("Initialized Groq fallback client")
        
        if not self.anthropic_client and not self.groq_client:
            logger.warning("No LLM clients available. Reasoning will be limited.")
    
    async def reason(
        self,
        query: str,
        context: List[Dict],
        sub_questions: Optional[List[Dict]] = None,
        depth: ReasoningDepth = ReasoningDepth.STANDARD,
        conversation_history: Optional[List[Dict]] = None
    ) -> ReasoningChain:
        """
        Perform reasoning on a query with given context.
        
        Args:
            query: The main query to answer
            context: Retrieved documents/chunks as context
            sub_questions: Optional list of sub-questions for multi-hop
            depth: Reasoning depth level
            conversation_history: Previous conversation for context
            
        Returns:
            ReasoningChain with complete reasoning steps
        """
        import time
        start_time = time.time()
        
        steps = []
        token_usage = {"input": 0, "output": 0}
        
        # Format context for the LLM
        context_text = self._format_context(context)
        
        if sub_questions and len(sub_questions) > 0:
            # Multi-hop reasoning with sub-questions
            previous_answers = []
            
            for i, sq in enumerate(sub_questions):
                step = await self._reason_step(
                    question=sq.get("question", sq) if isinstance(sq, dict) else sq,
                    context=context_text,
                    previous_answers=previous_answers,
                    step_number=i + 1,
                    depth=depth
                )
                steps.append(step)
                previous_answers.append({
                    "question": sq.get("question", sq) if isinstance(sq, dict) else sq,
                    "answer": step.answer
                })
                token_usage["input"] += step.metadata.get("input_tokens", 0)
                token_usage["output"] += step.metadata.get("output_tokens", 0)
            
            # Final synthesis step
            final_step = await self._synthesize_answer(
                query=query,
                context=context_text,
                reasoning_steps=steps,
                depth=depth
            )
            steps.append(final_step)
            final_answer = final_step.answer
            token_usage["input"] += final_step.metadata.get("input_tokens", 0)
            token_usage["output"] += final_step.metadata.get("output_tokens", 0)
        else:
            # Single-step reasoning with chain-of-thought
            step = await self._chain_of_thought(
                query=query,
                context=context_text,
                conversation_history=conversation_history,
                depth=depth
            )
            steps.append(step)
            final_answer = step.answer
            token_usage["input"] += step.metadata.get("input_tokens", 0)
            token_usage["output"] += step.metadata.get("output_tokens", 0)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Calculate overall confidence
        total_confidence = sum(s.confidence for s in steps) / len(steps) if steps else 0.0
        
        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=final_answer,
            total_confidence=total_confidence,
            reasoning_depth=depth,
            processing_time_ms=processing_time,
            token_usage=token_usage,
            metadata={
                "num_steps": len(steps),
                "context_chunks": len(context),
                "had_sub_questions": sub_questions is not None
            }
        )
    
    async def extract_relationships(
        self,
        context: List[Dict],
        focus_concepts: Optional[List[str]] = None
    ) -> List[Relationship]:
        """
        Extract relationships between concepts in the context.
        
        Args:
            context: Document chunks to analyze
            focus_concepts: Optional list of concepts to focus on
            
        Returns:
            List of discovered relationships
        """
        context_text = self._format_context(context)
        
        prompt = self.RELATIONSHIP_EXTRACTION_PROMPT
        if focus_concepts:
            prompt += f"\n\nFocus especially on relationships involving: {', '.join(focus_concepts)}"
        
        prompt += f"\n\nContext:\n{context_text}"
        
        response = await self._call_llm(prompt, system=prompt)
        
        # Parse JSON response
        relationships = []
        try:
            # Extract JSON from response
            json_match = self._extract_json(response)
            if json_match:
                rel_data = json.loads(json_match)
                for r in rel_data:
                    relationships.append(Relationship(
                        source=r.get("source", ""),
                        target=r.get("target", ""),
                        relationship_type=r.get("type", "relates-to"),
                        strength=float(r.get("strength", 0.5)),
                        evidence=[r.get("evidence", "")],
                        source_documents=self._extract_doc_names(context)
                    ))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse relationships JSON: {e}")
        
        return relationships
    
    async def find_implicit_information(
        self,
        context: List[Dict],
        query: Optional[str] = None
    ) -> List[ImplicitInfo]:
        """
        Find implicit information that can be derived from the context.
        
        Args:
            context: Document chunks to analyze
            query: Optional query to focus the analysis
            
        Returns:
            List of implicit information found
        """
        context_text = self._format_context(context)
        
        prompt = self.IMPLICIT_INFO_PROMPT
        if query:
            prompt += f"\n\nFocus on information relevant to: {query}"
        
        prompt += f"\n\nContext:\n{context_text}"
        
        response = await self._call_llm(prompt, system=prompt)
        
        # Parse JSON response
        implicit_info = []
        try:
            json_match = self._extract_json(response)
            if json_match:
                info_data = json.loads(json_match)
                for info in info_data:
                    implicit_info.append(ImplicitInfo(
                        statement=info.get("statement", ""),
                        derivation=info.get("derivation", ""),
                        supporting_facts=info.get("supporting_facts", []),
                        confidence=float(info.get("confidence", 0.5)),
                        source_documents=self._extract_doc_names(context)
                    ))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse implicit info JSON: {e}")
        
        return implicit_info
    
    async def _reason_step(
        self,
        question: str,
        context: str,
        previous_answers: List[Dict],
        step_number: int,
        depth: ReasoningDepth
    ) -> ReasoningStep:
        """Perform a single reasoning step."""
        prev_text = ""
        if previous_answers:
            prev_text = "\n".join([
                f"Q: {pa['question']}\nA: {pa['answer']}"
                for pa in previous_answers
            ])
        
        prompt = self.MULTI_HOP_PROMPT.format(
            context=context[:8000],  # Limit context size
            previous_answers=prev_text or "None yet",
            question=question
        )
        
        response = await self._call_llm(prompt, system=self.CHAIN_OF_THOUGHT_PROMPT)
        
        # Extract answer from response
        answer = self._extract_answer(response)
        
        return ReasoningStep(
            step_number=step_number,
            question=question,
            reasoning=response,
            answer=answer,
            confidence=self._estimate_confidence(response),
            metadata={
                "depth": depth.value,
                "input_tokens": len(prompt.split()) * 1.3,  # Rough estimate
                "output_tokens": len(response.split()) * 1.3
            }
        )
    
    async def _chain_of_thought(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]],
        depth: ReasoningDepth
    ) -> ReasoningStep:
        """Perform chain-of-thought reasoning."""
        # Build prompt with conversation context
        conv_text = ""
        if conversation_history:
            conv_text = "\n".join([
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
                for m in conversation_history[-6:]  # Last 6 messages
            ])
            conv_text = f"\nPrevious conversation:\n{conv_text}\n"
        
        # Adjust context size based on depth
        max_context = {
            ReasoningDepth.QUICK: 4000,
            ReasoningDepth.STANDARD: 8000,
            ReasoningDepth.DEEP: 16000,
            ReasoningDepth.EXHAUSTIVE: 32000
        }.get(depth, 8000)
        
        prompt = f"""Question: {query}
{conv_text}
Context from documents:
{context[:max_context]}

Think through this step by step, then provide a comprehensive answer."""

        response = await self._call_llm(prompt, system=self.CHAIN_OF_THOUGHT_PROMPT)
        answer = self._extract_answer(response)
        
        return ReasoningStep(
            step_number=1,
            question=query,
            reasoning=response,
            answer=answer,
            confidence=self._estimate_confidence(response),
            metadata={
                "depth": depth.value,
                "had_conversation_context": conversation_history is not None,
                "input_tokens": len(prompt.split()) * 1.3,
                "output_tokens": len(response.split()) * 1.3
            }
        )
    
    async def _synthesize_answer(
        self,
        query: str,
        context: str,
        reasoning_steps: List[ReasoningStep],
        depth: ReasoningDepth
    ) -> ReasoningStep:
        """Synthesize final answer from reasoning steps."""
        steps_text = "\n\n".join([
            f"**Step {s.step_number}: {s.question}**\nAnswer: {s.answer}"
            for s in reasoning_steps
        ])
        
        prompt = f"""Original question: {query}

Based on the following reasoning steps, provide a comprehensive final answer:

{steps_text}

Relevant context:
{context[:4000]}

Synthesize these findings into a clear, well-structured answer that:
1. Directly addresses the original question
2. Integrates insights from all reasoning steps
3. Cites specific evidence
4. Acknowledges any limitations or uncertainties"""

        response = await self._call_llm(prompt, system=self.CHAIN_OF_THOUGHT_PROMPT)
        answer = self._extract_answer(response)
        
        return ReasoningStep(
            step_number=len(reasoning_steps) + 1,
            question=f"Final synthesis for: {query}",
            reasoning=response,
            answer=answer,
            confidence=self._estimate_confidence(response),
            metadata={
                "is_synthesis": True,
                "num_steps_synthesized": len(reasoning_steps),
                "input_tokens": len(prompt.split()) * 1.3,
                "output_tokens": len(response.split()) * 1.3
            }
        )
    
    async def _call_llm(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> str:
        """Call the LLM (Claude or fallback)."""
        # Try Anthropic first
        if self.anthropic_client:
            try:
                message = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system or self.CHAIN_OF_THOUGHT_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic call failed: {e}, falling back to Groq")
        
        # Fallback to Groq
        if self.groq_client:
            try:
                full_prompt = f"{system}\n\n{prompt}" if system else prompt
                response = self.groq_client.invoke(full_prompt)
                return response.content
            except Exception as e:
                logger.error(f"Groq call failed: {e}")
                raise
        
        raise RuntimeError("No LLM client available for reasoning")
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context documents for LLM."""
        formatted = []
        for i, doc in enumerate(context):
            if isinstance(doc, dict):
                content = doc.get("page_content", doc.get("content", str(doc)))
                filename = doc.get("metadata", {}).get("filename", f"Document {i+1}")
                page = doc.get("metadata", {}).get("page", "")
            else:
                # LangChain Document object
                content = getattr(doc, "page_content", str(doc))
                metadata = getattr(doc, "metadata", {})
                filename = metadata.get("filename", f"Document {i+1}")
                page = metadata.get("page", "")
            
            page_str = f" (Page {page})" if page else ""
            formatted.append(f"[{filename}{page_str}]\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def _extract_doc_names(self, context: List[Dict]) -> List[str]:
        """Extract document names from context."""
        names = set()
        for doc in context:
            if isinstance(doc, dict):
                name = doc.get("metadata", {}).get("filename")
            else:
                name = getattr(doc, "metadata", {}).get("filename")
            if name:
                names.add(name)
        return list(names)
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer portion from a reasoning response."""
        # Look for explicit answer markers
        markers = ["**Answer**:", "Answer:", "**Final Answer**:", "In conclusion,", "Therefore,"]
        
        for marker in markers:
            if marker in response:
                idx = response.index(marker)
                answer = response[idx + len(marker):].strip()
                # Take until the next section or end
                for end_marker in ["\n\n**", "\n\n##"]:
                    if end_marker in answer:
                        answer = answer[:answer.index(end_marker)]
                return answer.strip()
        
        # If no marker, return last paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        return response.strip()
    
    def _extract_json(self, response: str) -> Optional[str]:
        """Extract JSON from a response."""
        import re
        # Try to find JSON array
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            return json_match.group()
        return None
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on response characteristics."""
        confidence = 0.7  # Base confidence
        
        # Lower confidence for uncertainty markers
        uncertainty_markers = [
            "i'm not sure", "it's unclear", "may", "might", "possibly",
            "could be", "uncertain", "limited information", "not enough"
        ]
        response_lower = response.lower()
        for marker in uncertainty_markers:
            if marker in response_lower:
                confidence -= 0.1
        
        # Higher confidence for evidence markers
        evidence_markers = [
            "according to", "the document states", "specifically",
            "clearly", "evidence shows", "based on"
        ]
        for marker in evidence_markers:
            if marker in response_lower:
                confidence += 0.05
        
        return max(0.1, min(1.0, confidence))


# Global instance
_reasoner = None


def get_semantic_reasoner(**kwargs) -> SemanticReasoner:
    """Get the global semantic reasoner instance."""
    global _reasoner
    if _reasoner is None:
        _reasoner = SemanticReasoner(**kwargs)
    return _reasoner
