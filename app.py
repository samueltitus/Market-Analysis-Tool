import os
import json
import time
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# --- Core Gemini call with fallback and retry ---
def call_gemini(prompt, retries=1):
    models = ["gemini-2.5-flash", "gemini-2.0-flash"]
    last_error = None
    for model in models:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )
                sources = []
                candidate = response.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    chunks = candidate.grounding_metadata.grounding_chunks or []
                    for chunk in chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            sources.append({
                                "title": chunk.web.title,
                                "url": chunk.web.uri
                            })
                return response.text, sources
            except Exception as e:
                last_error = e
                if "503" in str(e) or "429" in str(e):
                    time.sleep(60)
                    continue
                elif attempt < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    break
    raise last_error


def parse_json(text):
    text = text.strip()
    # Strip markdown code block if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            if part.startswith("{"):
                text = part
                break
    # Find the outermost JSON object in case there's extra text around it
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


# --- Section fetchers ---
def fetch_market_overview(startup_name, industry, description):
    prompt = f"""
You are a market research analyst. Search Google for real data about the {industry} market.

Startup: {startup_name}
Description: {description}

Find real sourced figures. Use null if you cannot find a specific figure. Return ONLY raw JSON, no markdown.

{{
  "market_size_current": "e.g. $12.06 billion (2024)",
  "market_size_projected": "e.g. $47.82 billion (2030)",
  "cagr": "e.g. 25.8%",
  "key_drivers": ["specific driver with real evidence", "specific driver with real evidence"],
  "market_stage": "emerging | growing | mature",
  "geographic_focus": "e.g. North America leads with X% share",
  "summary": "2-3 punchy sentences covering size, growth rate, and the single most important dynamic an investor needs to know"
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_market_sizing(startup_name, industry, description):
    prompt = f"""
You are a market research analyst. Search Google for real TAM/SAM/SOM figures for this startup.

Startup: {startup_name}
Industry: {industry}
Description: {description}

TAM = total global market for this category
SAM = portion realistically reachable (by geography, segment, or business model)
SOM = realistic capture in first 3-5 years based on comparable startups

Return ONLY raw JSON, no markdown. Use null if you cannot find real figures.

{{
  "tam_billion": 12.5,
  "tam_source_note": "e.g. Grand View Research, 2024",
  "sam_billion": 2.1,
  "sam_reasoning": "e.g. Focused on English-speaking SMBs in North America and Europe",
  "som_billion": 0.05,
  "som_reasoning": "e.g. 2.5% capture in 5 years is realistic based on comparable early-stage SaaS companies"
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_competitors(startup_name, industry, description):
    prompt = f"""
You are a competitive intelligence analyst. Search Google to find real competitors in the {industry} market.

Startup: {startup_name}
Description: {description}

Find 4-6 REAL competitors. Do NOT mention {startup_name} itself. Find their actual website URLs.

Return ONLY raw JSON, no markdown.

{{
  "competitors": [
    {{
      "name": "Competitor Name",
      "website": "https://example.com",
      "founded": 2018,
      "funding": "$10M Series A",
      "focus": "One sentence on what they do and who they serve",
      "differentiator": "What makes them stand out from others"
    }}
  ],
  "landscape_summary": "1-2 sentences on the overall competitive dynamic — fragmented, dominated by a few players, consolidating, etc."
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_target_customer(startup_name, industry, description):
    prompt = f"""
You are a market research analyst. Search Google for real data about the target customer for this startup.

Startup: {startup_name}
Industry: {industry}
Description: {description}

Find real research, studies, or reports. Every point should be based on real evidence.

Return ONLY raw JSON, no markdown. Use null if not found.

{{
  "customer_profile": "1-2 sentences with specific details — who they are, company size, role, demographics",
  "key_problems": ["specific problem with real evidence or stat", "specific problem with real evidence or stat"],
  "current_solutions": "How they currently solve this problem and why those solutions fall short",
  "willingness_to_pay": "Real evidence of budget or spend in this area — contracts, surveys, or market data",
  "market_evidence": "One specific stat or finding from real research that validates this customer exists and has this problem"
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_key_risks(startup_name, industry, description):
    prompt = f"""
You are a startup risk analyst. Search Google for real risks facing companies in the {industry} space.

Startup: {startup_name}
Description: {description}

Find 3-4 real, specific risks that existing competitors or similar startups have faced. Link each to a real example — a news article, study, or documented case.

Return ONLY raw JSON, no markdown.

{{
  "risks": [
    {{
      "risk": "Short risk title",
      "explanation": "1-2 sentences explaining why this is a real risk with specific context",
      "example": "Real example: e.g. Company X faced this issue in 2023 when..."
    }}
  ]
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_swot(startup_name, industry, description):
    prompt = f"""
You are a strategic analyst. Search Google to build a sourced SWOT analysis for this startup.

Startup: {startup_name}
Industry: {industry}
Description: {description}

Every point must be specific and based on real market evidence. No generic statements.

Return ONLY raw JSON, no markdown.

{{
  "strengths": ["specific strength backed by real market evidence"],
  "weaknesses": ["specific weakness with real market context"],
  "opportunities": ["specific opportunity with a real market stat or trend"],
  "threats": ["specific threat with a real example or data point"]
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_idea_originality(startup_name, industry, description):
    prompt = f"""
You are a startup analyst. Search Google to assess how original this startup idea is and whether it can realistically survive.

Startup: {startup_name}
Industry: {industry}
Description: {description}

Search for existing solutions and similar startups. Be honest — if this space is already well-served, say so clearly.

Return ONLY raw JSON, no markdown.

{{
  "novelty": "existing | partial | novel",
  "why_it_could_survive": "1-2 sentences with specific market evidence for a viable path",
  "why_it_might_not": "1-2 sentences with specific competitive threats or conditions that could kill it",
  "verdict": "1 honest sentence summarising the originality assessment"
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


def fetch_opportunity_score(startup_name, industry, description, all_data):
    summary = f"""
Market size: {all_data['market_overview'].get('market_size_current')} growing to {all_data['market_overview'].get('market_size_projected')}
CAGR: {all_data['market_overview'].get('cagr')}
Market stage: {all_data['market_overview'].get('market_stage')}
Competitors found: {len(all_data['competitors'].get('competitors', []))}
Novelty: {all_data['originality'].get('novelty')}
Verdict: {all_data['originality'].get('verdict')}
Key risks: {len(all_data['risks'].get('risks', []))}
"""
    prompt = f"""
You are a startup investment analyst. Based on the following research, score this startup's market opportunity.

Startup: {startup_name}
Industry: {industry}
Description: {description}

Research summary:
{summary}

Score 1-10:
1-3 = Poor (saturated, shrinking, or no clear path)
4-6 = Moderate (viable but competitive or uncertain)
7-8 = Strong (growing market, clear gap, realistic path)
9-10 = Exceptional (only if all signals are very strong)

Return ONLY raw JSON, no markdown.

{{
  "score": 7,
  "reasoning": "2-3 sentences recapping the key findings — market size, competition, originality, and main risk — that drove this score"
}}
"""
    text, sources = call_gemini(prompt)
    data = parse_json(text)
    data["sources"] = sources
    return data


# --- Chart and display helpers ---
def render_tam_chart(sizing):
    tam = sizing.get("tam_billion")
    sam = sizing.get("sam_billion")
    som = sizing.get("som_billion")
    if not all([tam, sam, som]):
        return None
    values = [tam, sam, som]
    values_sorted = sorted(values, reverse=True)
    fig = go.Figure(go.Funnel(
        y=["Total Addressable Market (TAM)", "Serviceable Addressable Market (SAM)", "Serviceable Obtainable Market (SOM)"],
        x=values_sorted,
        textinfo="value+percent initial",
        texttemplate="%{value}B USD",
        marker={"color": ["#1f77b4", "#2ca02c", "#ff7f0e"]}
    ))
    fig.update_layout(title="Market Sizing (USD Billions)", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def render_swot(swot):
    cols = st.columns(2)
    quadrants = [
        ("Strengths", swot.get("strengths", []), "#d4edda", "black"),
        ("Weaknesses", swot.get("weaknesses", []), "#f8d7da", "black"),
        ("Opportunities", swot.get("opportunities", []), "#d1ecf1", "black"),
        ("Threats", swot.get("threats", []), "#fff3cd", "black"),
    ]
    for i, (label, items, bg, fg) in enumerate(quadrants):
        with cols[i % 2]:
            items_html = "".join(f"<li>{item}</li>" for item in items)
            st.markdown(
                f"""<div style="background:{bg};padding:14px;border-radius:8px;margin-bottom:10px;color:{fg}">
                <strong>{label}</strong><ul style="margin:6px 0 0 0">{items_html}</ul></div>""",
                unsafe_allow_html=True
            )


def show_sources(sources):
    if sources:
        st.markdown("---")
        st.caption("Sources")
        for i, source in enumerate(sources, start=1):
            st.markdown(f"*{i}. [{source['title']}]({source['url']})*")


# --- App layout ---
st.set_page_config(page_title="Market Analysis Tool", layout="wide")
st.title("Market Analysis Tool")
st.write("Enter startup details to generate an investor-ready market analysis.")

with st.form("analysis_form"):
    startup_name = st.text_input("Startup Name", placeholder="e.g. HealthTrack")
    industry = st.text_input("Industry / Sector", placeholder="e.g. Digital Health, Fintech")
    description = st.text_area("Brief Description", placeholder="What does the startup do and what problem does it solve?", height=130)
    submitted = st.form_submit_button("Generate Analysis", type="primary")

if submitted:
    if startup_name and industry and description:
        progress = st.progress(0, text="Starting analysis...")
        try:
            progress.progress(10, text="Running analysis in parallel...")

            args = (startup_name, industry, description)
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_overview    = executor.submit(fetch_market_overview,  *args)
                f_sizing      = executor.submit(fetch_market_sizing,    *args)
                f_competitors = executor.submit(fetch_competitors,      *args)
                f_customer    = executor.submit(fetch_target_customer,  *args)
                f_risks       = executor.submit(fetch_key_risks,        *args)
                f_swot        = executor.submit(fetch_swot,             *args)
                f_originality = executor.submit(fetch_idea_originality, *args)

            market_overview = f_overview.result()
            market_sizing   = f_sizing.result()
            competitors     = f_competitors.result()
            target_customer = f_customer.result()
            key_risks       = f_risks.result()
            swot            = f_swot.result()
            originality     = f_originality.result()

            progress.progress(80, text="Scoring opportunity...")

            all_data = {
                "market_overview": market_overview,
                "sizing":          market_sizing,
                "competitors":     competitors,
                "customer":        target_customer,
                "risks":           key_risks,
                "swot":            swot,
                "originality":     originality
            }

            opportunity = fetch_opportunity_score(startup_name, industry, description, all_data)

            progress.progress(100, text="Complete!")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

        progress.empty()
        st.success("Analysis complete!")

        # --- Key metrics row ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Opportunity Score", f"{opportunity.get('score', 'N/A')} / 10")
        m2.metric("Competitors Found", len(competitors.get("competitors", [])))
        m3.metric("Market CAGR", market_overview.get("cagr", "N/A"))
        m4.metric("Idea Novelty", f"{novelty_icon} {originality.get('novelty', 'N/A').capitalize()}")

        st.divider()

        # --- Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market Overview", "Competition", "Customer & Risks", "SWOT", "Summary"])

        with tab1:
            st.subheader("Market Overview")
            st.write(market_overview.get("summary"))

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Market Size", market_overview.get("market_size_current", "N/A"))
            col2.metric("Projected Size", market_overview.get("market_size_projected", "N/A"))
            col3.metric("Growth Rate (CAGR)", market_overview.get("cagr", "N/A"))

            st.subheader("Key Market Drivers")
            for driver in market_overview.get("key_drivers", []):
                st.markdown(f"- {driver}")

            show_sources(market_overview.get("sources", []))

            st.divider()
            st.subheader("Market Sizing")

            tam_chart = render_tam_chart(market_sizing)
            if tam_chart:
                st.plotly_chart(tam_chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**SAM Reasoning:** {market_sizing.get('sam_reasoning', 'N/A')}")
            with col2:
                st.markdown(f"**SOM Reasoning:** {market_sizing.get('som_reasoning', 'N/A')}")

            show_sources(market_sizing.get("sources", []))

        with tab2:
            st.subheader("Competitive Landscape")
            st.write(competitors.get("landscape_summary"))

            st.subheader("Key Competitors")
            for comp in competitors.get("competitors", []):
                with st.expander(f"**{comp.get('name')}** — {comp.get('focus', '')}"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Founded:** {comp.get('founded', 'N/A')}")
                    c2.markdown(f"**Funding:** {comp.get('funding', 'N/A')}")
                    st.markdown(f"**Differentiator:** {comp.get('differentiator', 'N/A')}")
                    if comp.get("website"):
                        st.markdown(f"[Visit website]({comp.get('website')})")

            show_sources(competitors.get("sources", []))

            st.divider()
            st.subheader("Idea Originality")
            st.markdown(f"**Verdict:** {verdict_colors.get(originality.get('novelty', 'partial'))} {originality.get('verdict', '')}")
            st.markdown(f"**Why it could survive:** {originality.get('why_it_could_survive', 'N/A')}")
            st.markdown(f"**Why it might not:** {originality.get('why_it_might_not', 'N/A')}")
            show_sources(originality.get("sources", []))

        with tab3:
            st.subheader("Target Customer")
            st.write(target_customer.get("customer_profile"))

            st.subheader("Key Problems They Face")
            for problem in target_customer.get("key_problems", []):
                st.markdown(f"- {problem}")

            st.subheader("Current Solutions")
            st.write(target_customer.get("current_solutions", "N/A"))

            st.subheader("Willingness to Pay")
            st.write(target_customer.get("willingness_to_pay", "N/A"))

            show_sources(target_customer.get("sources", []))

            st.divider()
            st.subheader("Key Risks")
            for risk in key_risks.get("risks", []):
                with st.expander(f"⚠️ {risk.get('risk', '')}"):
                    st.write(risk.get("explanation", ""))
                    if risk.get("example"):
                        st.markdown(f"*Example: {risk.get('example')}*")

            show_sources(key_risks.get("sources", []))

        with tab4:
            st.subheader("SWOT Analysis")
            render_swot(swot)
            show_sources(swot.get("sources", []))

        with tab5:
            st.subheader("Opportunity Score")
            score = opportunity.get("score", 0)
            st.markdown(f"### {score} / 10")
            st.progress(int(score) / 10)
            st.write(opportunity.get("reasoning", ""))
            show_sources(opportunity.get("sources", []))

    else:
        st.warning("Please fill in all three fields before generating.")
