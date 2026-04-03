"""
KOS Research Engine
===================
Gives the organism the ability to research any topic using the internet.
Combines web search + web fetch + knowledge extraction + synthesis.

Workflow:
1. Receive research topic
2. Generate search queries (multiple angles)
3. Fetch and extract key findings
4. Synthesize into actionable knowledge
5. Feed back into organism's memory + material search

Safety: All research passes through First Law filter.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ResearchFinding:
    """A single finding from web research."""
    query: str
    source_url: str
    title: str
    snippet: str
    relevance_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchReport:
    """Complete research report on a topic."""
    topic: str
    queries_used: List[str]
    findings: List[ResearchFinding]
    synthesis: str
    key_facts: List[str]
    actionable_insights: List[str]
    suggested_compositions: List[Dict] = field(default_factory=list)
    total_sources: int = 0
    research_time_seconds: float = 0.0
    safety_cleared: bool = True


# Safety keywords that block research
BLOCKED_TOPICS = [
    "weapon", "explosive", "bomb", "poison", "toxin",
    "biological weapon", "chemical weapon", "nerve agent",
    "how to kill", "how to harm", "assassination",
    "malware", "ransomware", "exploit", "hack password",
]


class ResearchEngine:
    """
    Internet research capability for the KOS organism.

    Uses web search APIs to find information on any topic,
    then synthesizes findings into actionable knowledge.
    """

    def __init__(self, first_law: str = "DO NOT HARM A HUMAN BEING"):
        self.first_law = first_law
        self.research_history: List[ResearchReport] = []
        self.knowledge_base: Dict[str, List[str]] = {}  # topic -> facts

    def _safety_check(self, topic: str) -> Tuple[bool, str]:
        """Check if research topic passes safety filter."""
        topic_lower = topic.lower()
        for blocked in BLOCKED_TOPICS:
            if blocked in topic_lower:
                return False, f"BLOCKED: Research on '{blocked}' violates First Law: {self.first_law}"
        return True, "OK"

    def _generate_queries(self, topic: str, n_queries: int = 5) -> List[str]:
        """Generate diverse search queries for a topic."""
        queries = [topic]

        # Add specialized query variants
        topic_words = topic.split()

        # Scientific query
        queries.append(f"{topic} research paper findings 2024 2025")

        # Recent developments
        queries.append(f"{topic} latest developments breakthrough")

        # Practical applications
        queries.append(f"{topic} practical applications real world")

        # Technical details
        queries.append(f"{topic} technical specifications data properties")

        # Safety / environmental
        queries.append(f"{topic} safety environmental impact toxicity")

        # Alternatives / improvements
        queries.append(f"{topic} alternatives improvements optimization")

        return queries[:n_queries]

    def _web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform a web search using DuckDuckGo HTML API (no key needed).

        Returns list of {title, url, snippet}.
        """
        results = []
        try:
            encoded_q = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_q}"

            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (KOS-Organism Research Engine)"
            })

            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="replace")

            # Parse results from DuckDuckGo HTML
            # Look for result blocks
            result_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                re.DOTALL
            )

            for match in result_pattern.finditer(html):
                if len(results) >= max_results:
                    break
                href = match.group(1)
                title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                snippet = re.sub(r'<[^>]+>', '', match.group(3)).strip()

                # Decode DuckDuckGo redirect URL
                if "uddg=" in href:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]

                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                })

            # Fallback: simpler parsing if structured parsing fails
            if not results:
                # Try simpler pattern
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>',
                    html, re.DOTALL
                )
                links = re.findall(
                    r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                    html, re.DOTALL
                )
                for i, (href, title) in enumerate(links[:max_results]):
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet,
                    })

        except Exception as e:
            results.append({
                "title": f"Search error",
                "url": "",
                "snippet": f"Could not search: {str(e)}",
            })

        return results

    def _fetch_page_text(self, url: str, max_chars: int = 5000) -> str:
        """Fetch a web page and extract text content."""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (KOS-Organism Research Engine)"
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="replace")

            # Strip HTML tags and scripts
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:max_chars]

        except Exception as e:
            return f"Could not fetch page: {str(e)}"

    def _extract_key_facts(self, text: str, topic: str) -> List[str]:
        """Extract key facts from text related to topic."""
        facts = []
        sentences = re.split(r'[.!?]+', text)
        topic_words = set(topic.lower().split())

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue
            sent_lower = sent.lower()
            # Score by topic word overlap
            overlap = sum(1 for w in topic_words if w in sent_lower)
            if overlap >= 1:
                # Check for factual indicators
                factual_words = [
                    "is", "are", "was", "has", "have", "shows", "found",
                    "demonstrated", "achieved", "efficiency", "bandgap",
                    "eV", "percent", "%", "temperature", "stable",
                    "toxic", "safe", "material", "compound",
                ]
                if any(fw in sent_lower for fw in factual_words):
                    facts.append(sent.strip())

        # Deduplicate and limit
        seen = set()
        unique_facts = []
        for f in facts:
            key = f[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        return unique_facts[:20]

    def research(self, topic: str, depth: str = "standard") -> ResearchReport:
        """Conduct research on a topic.

        Args:
            topic: What to research
            depth: "quick" (2 queries), "standard" (5), "deep" (8)

        Returns:
            ResearchReport with findings and synthesis
        """
        t0 = time.time()

        # Safety check
        safe, msg = self._safety_check(topic)
        if not safe:
            return ResearchReport(
                topic=topic,
                queries_used=[],
                findings=[],
                synthesis=msg,
                key_facts=[],
                actionable_insights=[msg],
                safety_cleared=False,
                research_time_seconds=time.time() - t0,
            )

        # Generate queries
        n_queries = {"quick": 2, "standard": 5, "deep": 8}.get(depth, 5)
        queries = self._generate_queries(topic, n_queries)

        # Search and collect findings
        all_findings: List[ResearchFinding] = []
        all_facts: List[str] = []

        for query in queries:
            search_results = self._web_search(query)

            for sr in search_results:
                finding = ResearchFinding(
                    query=query,
                    source_url=sr["url"],
                    title=sr["title"],
                    snippet=sr["snippet"],
                    relevance_score=0.5,
                )
                all_findings.append(finding)

                # Extract facts from snippets
                if sr["snippet"]:
                    facts = self._extract_key_facts(sr["snippet"], topic)
                    all_facts.extend(facts)

            # For deep research, also fetch top result pages
            if depth == "deep" and search_results:
                for sr in search_results[:2]:
                    if sr["url"]:
                        page_text = self._fetch_page_text(sr["url"])
                        facts = self._extract_key_facts(page_text, topic)
                        all_facts.extend(facts)

        # Deduplicate facts
        seen = set()
        unique_facts = []
        for f in all_facts:
            key = f[:40].lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        # Generate synthesis
        synthesis = self._synthesize(topic, unique_facts, all_findings)

        # Generate actionable insights
        insights = self._generate_insights(topic, unique_facts)

        # If this is a materials topic, suggest compositions
        compositions = self._suggest_compositions(topic, unique_facts)

        # Store in knowledge base
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].extend(unique_facts[:10])

        report = ResearchReport(
            topic=topic,
            queries_used=queries,
            findings=all_findings,
            synthesis=synthesis,
            key_facts=unique_facts[:30],
            actionable_insights=insights,
            suggested_compositions=compositions,
            total_sources=len(all_findings),
            research_time_seconds=time.time() - t0,
        )

        self.research_history.append(report)
        return report

    def _synthesize(self, topic: str, facts: List[str],
                    findings: List[ResearchFinding]) -> str:
        """Synthesize findings into a coherent summary."""
        if not facts and not findings:
            return f"No significant findings for '{topic}'. Consider broadening the search terms."

        parts = [f"Research synthesis for: {topic}"]
        parts.append(f"Analyzed {len(findings)} sources, extracted {len(facts)} key facts.")

        if facts:
            parts.append("\nKey findings:")
            for i, fact in enumerate(facts[:10], 1):
                parts.append(f"  {i}. {fact}")

        # Identify patterns
        topic_lower = topic.lower()
        if "solar" in topic_lower or "perovskite" in topic_lower:
            parts.append("\nMaterial science context: Research focuses on photovoltaic "
                        "applications. Key metrics are bandgap (optimal 1.1-1.5 eV), "
                        "stability, and toxicity.")
        elif "battery" in topic_lower:
            parts.append("\nEnergy storage context: Key metrics are energy density, "
                        "cycle life, safety, and cost.")
        elif "catalyst" in topic_lower:
            parts.append("\nCatalysis context: Key metrics are selectivity, turnover "
                        "frequency, stability, and cost of materials.")

        return "\n".join(parts)

    def _generate_insights(self, topic: str, facts: List[str]) -> List[str]:
        """Generate actionable insights from facts."""
        insights = []

        # Material-related insights
        material_keywords = ["efficiency", "bandgap", "stable", "toxic", "cost",
                           "performance", "degradation", "lifetime"]
        for fact in facts[:15]:
            fact_lower = fact.lower()
            for kw in material_keywords:
                if kw in fact_lower:
                    insights.append(f"[{kw.upper()}] {fact}")
                    break

        if not insights:
            insights.append(f"Research on '{topic}' requires deeper investigation. "
                          "Consider using 'deep' mode for more comprehensive results.")

        return insights[:10]

    def _suggest_compositions(self, topic: str, facts: List[str]) -> List[Dict]:
        """If the research is about materials, suggest compositions to simulate."""
        compositions = []
        topic_lower = topic.lower()

        # Extract any chemical formulas mentioned
        formula_pattern = re.compile(r'\b([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)\b')

        for fact in facts:
            for match in formula_pattern.finditer(fact):
                groups = match.groups()
                comp = {}
                for i in range(0, len(groups), 2):
                    elem = groups[i]
                    count = int(groups[i+1]) if groups[i+1] else 1
                    if elem and len(elem) <= 2:
                        comp[elem] = count
                if len(comp) >= 2:
                    compositions.append({
                        "formula": match.group(0),
                        "composition": comp,
                        "source": fact[:80],
                    })

        # Default suggestions based on topic
        if "solar" in topic_lower or "perovskite" in topic_lower:
            compositions.extend([
                {"formula": "CsSnI3", "composition": {"Cs": 1, "Sn": 1, "I": 3},
                 "source": "Lead-free perovskite candidate"},
                {"formula": "CsGeI3", "composition": {"Cs": 1, "Ge": 1, "I": 3},
                 "source": "Germanium-based lead-free perovskite"},
                {"formula": "CsBi3I10", "composition": {"Cs": 1, "Bi": 3, "I": 10},
                 "source": "Bismuth-based non-toxic alternative"},
            ])

        return compositions[:10]

    def quick_search(self, query: str) -> List[Dict]:
        """Quick search without full research synthesis."""
        safe, msg = self._safety_check(query)
        if not safe:
            return [{"error": msg}]
        return self._web_search(query, max_results=10)

    def get_knowledge(self, topic: str) -> List[str]:
        """Retrieve accumulated knowledge on a topic."""
        # Exact match
        if topic in self.knowledge_base:
            return self.knowledge_base[topic]

        # Partial match
        topic_lower = topic.lower()
        results = []
        for key, facts in self.knowledge_base.items():
            if topic_lower in key.lower() or key.lower() in topic_lower:
                results.extend(facts)

        return results
