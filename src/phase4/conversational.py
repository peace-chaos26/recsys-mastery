"""
conversational.py
-----------------
Conversational recommendation agent using LLMs.

Instead of mining interaction history, this agent:
  1. Asks the user what they're looking for (dialogue)
  2. Extracts structured preferences from natural language
  3. Retrieves semantically matching items
  4. Explains recommendations in natural language

Zero cold-start: works for brand new users with no history.
Zero interaction data: relies entirely on expressed preferences.

This is the architecture behind:
  - Amazon Rufus (LLM shopping assistant)
  - Spotify DJ (natural language music curation)
  - Netflix "find me something like X" feature

Uses OpenAI gpt-4o-mini for dialogue and preference extraction.
Falls back to rule-based mode if no API key is set.
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 1. User preference state
# --------------------------------------------------------------------------

@dataclass
class UserPreferences:
    """
    Structured preferences extracted from dialogue.

    Built up incrementally as the conversation progresses.
    Each turn potentially adds or refines preferences.
    """
    occasion      : str = ""          # birthday, holiday, self-use
    recipient     : str = ""          # friend, parent, colleague, self
    budget_min    : float = 0.0       # minimum gift value
    budget_max    : float = 999.0     # maximum gift value
    interests     : list = field(default_factory=list)   # gaming, music, etc.
    excluded      : list = field(default_factory=list)   # brands/types to avoid
    preferred_brands: list = field(default_factory=list) # preferred brands
    raw_query     : str = ""          # original natural language query

    def to_search_query(self) -> str:
        """Convert preferences to a search query for semantic retrieval."""
        parts = []
        if self.occasion:
            parts.append(f"{self.occasion} gift")
        if self.recipient:
            parts.append(f"for {self.recipient}")
        if self.interests:
            parts.append(" ".join(self.interests))
        if self.preferred_brands:
            parts.append(" ".join(self.preferred_brands))
        if self.budget_max < 999:
            parts.append(f"${self.budget_max:.0f} value")
        return " ".join(parts) if parts else self.raw_query

    def is_complete(self) -> bool:
        """Do we have enough info to make recommendations?"""
        return bool(self.raw_query or self.occasion or self.interests
                    or self.preferred_brands)


# --------------------------------------------------------------------------
# 2. Preference extractor
# --------------------------------------------------------------------------

def extract_preferences(user_message: str,
                         existing_prefs: UserPreferences,
                         client) -> UserPreferences:
    """
    Extract structured preferences from a natural language message.

    Uses LLM to parse the user's intent and merge with existing prefs.
    Returns updated UserPreferences.
    """
    existing_json = json.dumps({
        "occasion": existing_prefs.occasion,
        "recipient": existing_prefs.recipient,
        "budget_min": existing_prefs.budget_min,
        "budget_max": existing_prefs.budget_max,
        "interests": existing_prefs.interests,
        "excluded": existing_prefs.excluded,
        "preferred_brands": existing_prefs.preferred_brands,
    })

    prompt = f"""Extract gift card preferences from this message.
Merge with existing preferences (don't overwrite unless explicitly changed).

Existing preferences: {existing_json}

New message: "{user_message}"

Return ONLY valid JSON with these fields:
{{
  "occasion": "birthday|holiday|graduation|thank_you|self|other|",
  "recipient": "friend|parent|colleague|child|partner|self|",
  "budget_min": 0,
  "budget_max": 999,
  "interests": ["gaming", "music", "streaming", "shopping", "food"],
  "excluded": [],
  "preferred_brands": ["Amazon", "Netflix", "Steam", etc]
}}

Use empty string for unknown fields. Keep existing values if not mentioned."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return UserPreferences(
            occasion=data.get("occasion", existing_prefs.occasion),
            recipient=data.get("recipient", existing_prefs.recipient),
            budget_min=float(data.get("budget_min", existing_prefs.budget_min)),
            budget_max=float(data.get("budget_max", existing_prefs.budget_max)),
            interests=data.get("interests", existing_prefs.interests),
            excluded=data.get("excluded", existing_prefs.excluded),
            preferred_brands=data.get("preferred_brands",
                                       existing_prefs.preferred_brands),
            raw_query=user_message,
        )
    except Exception as e:
        print(f"[conv] Preference extraction error: {e}")
        existing_prefs.raw_query = user_message
        return existing_prefs


# --------------------------------------------------------------------------
# 3. Recommendation generator
# --------------------------------------------------------------------------

def generate_recommendations(
    preferences: UserPreferences,
    retriever,
    descriptions: dict,
    n: int = 5,
) -> list[tuple[str, str, float]]:
    """
    Retrieve items matching the user's expressed preferences.

    Uses semantic retrieval against a synthetic user profile
    built from the preference query.

    Returns list of (item_id, description, score) tuples.
    """
    query = preferences.to_search_query()
    if not query:
        return []

    # Build a synthetic "user history" by finding items matching preferences
    # In production: embed the preference query directly with the LLM encoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    # Find items that match preference keywords
    query_terms = query.lower().split()
    matching_items = []

    for item_id, desc in descriptions.items():
        desc_lower = desc.lower()
        # Score by keyword overlap
        score = sum(1.0 for term in query_terms if term in desc_lower)
        # Budget filter
        import re
        amounts = re.findall(r'\$(\d+)', desc)
        if amounts:
            amount = float(amounts[0])
            if preferences.budget_min <= amount <= preferences.budget_max:
                score += 0.5  # bonus for budget match
        # Brand preference boost
        for brand in preferences.preferred_brands:
            if brand.lower() in desc_lower:
                score += 1.0
        # Exclusion filter
        excluded = False
        for excl in preferences.excluded:
            if excl.lower() in desc_lower:
                excluded = True
                break
        if not excluded and score > 0:
            matching_items.append((item_id, desc, score))

    # Sort by score and return top-N
    matching_items.sort(key=lambda x: x[2], reverse=True)

    # Add semantic retrieval if retriever is available
    if retriever and hasattr(retriever, 'embeddings'):
        # Use the first matching item as a "seed" for semantic expansion
        if matching_items:
            seed_id = matching_items[0][0]
            if seed_id in retriever.item2idx:
                idx = retriever.item2idx[seed_id]
                seed_vec = retriever.embeddings[idx:idx+1]
                from sklearn.preprocessing import normalize
                seed_vec = normalize(seed_vec, norm="l2").astype(np.float32)
                distances, indices = retriever.index.search(seed_vec, n * 3)
                for ann_idx, dist in zip(indices[0], distances[0]):
                    if ann_idx >= 0:
                        ann_item = retriever.item_ids[ann_idx]
                        ann_desc = descriptions.get(ann_item, "")
                        # Check not already in matching and not excluded
                        excluded = any(e.lower() in ann_desc.lower()
                                       for e in preferences.excluded)
                        if (not excluded and
                            not any(m[0] == ann_item for m in matching_items[:n])):
                            matching_items.append(
                                (ann_item, ann_desc, float(dist) * 0.8)
                            )

    return matching_items[:n]


# --------------------------------------------------------------------------
# 4. Response generator
# --------------------------------------------------------------------------

def generate_response(
    turn: int,
    user_message: str,
    preferences: UserPreferences,
    recommendations: list,
    descriptions: dict,
    client,
    conversation_history: list,
) -> str:
    """
    Generate a natural language response with recommendations.

    Handles two modes:
      - Clarification: ask follow-up questions if preferences incomplete
      - Recommendation: present curated items with explanations
    """
    if not preferences.is_complete() and turn < 2:
        # Ask a clarifying question
        prompt = f"""You are a helpful gift card recommendation assistant.
The user said: "{user_message}"

Ask ONE natural follow-up question to better understand their needs.
Be conversational and brief (1-2 sentences max).
Focus on: occasion, recipient, budget, or interests — whichever is most unclear."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=conversation_history + [
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ("Could you tell me a bit more? "
                    "Who is this gift for and what's your budget?")

    # Generate recommendation response
    if not recommendations:
        return ("I couldn't find specific matches, but popular options include "
                "Amazon, Netflix, and Spotify gift cards. "
                "Can you tell me more about the recipient's interests?")

    rec_list = "\n".join(
        f"- {desc} (relevance: {score:.2f})"
        for _, desc, score in recommendations[:3]
    )

    context = preferences.to_search_query()
    prompt = f"""You are a helpful gift card recommendation assistant.

User is looking for: {context}

Top recommendations:
{rec_list}

Write a brief, friendly response (3-4 sentences) that:
1. Acknowledges what they're looking for
2. Presents the top 2-3 recommendations naturally
3. Explains briefly why each fits their needs
Do NOT use bullet points. Be conversational."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=conversation_history + [
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[conv] Response generation error: {e}")
        rec_names = [desc.split(" - ")[0] for _, desc, _ in recommendations[:3]]
        return f"Based on your preferences, I'd suggest: {', '.join(rec_names)}."


# --------------------------------------------------------------------------
# 5. Conversational agent
# --------------------------------------------------------------------------

class ConversationalRecommender:
    """
    Multi-turn conversational recommendation agent.

    Maintains conversation history and preference state across turns.
    Each turn: extract preferences → retrieve items → generate response.

    This is stateful — call chat() repeatedly for multi-turn dialogue.
    """

    def __init__(self, retriever, descriptions: dict, client):
        self.retriever    = retriever
        self.descriptions = descriptions
        self.client       = client
        self.preferences  = UserPreferences()
        self.history      = [
            {"role": "system",
             "content": ("You are a helpful, friendly gift card recommendation "
                         "assistant. Help users find the perfect gift card "
                         "by understanding their needs through natural conversation.")}
        ]
        self.turn = 0
        self.last_recommendations = []

    def chat(self, user_message: str) -> str:
        """
        Process one turn of conversation.
        Returns the agent's response as a string.
        """
        self.turn += 1
        self.history.append({"role": "user", "content": user_message})

        # Extract preferences from this message
        self.preferences = extract_preferences(
            user_message, self.preferences, self.client
        )

        # Retrieve recommendations if we have enough info
        recs = []
        if self.preferences.is_complete():
            recs = generate_recommendations(
                self.preferences, self.retriever,
                self.descriptions, n=5
            )
            self.last_recommendations = recs

        # Generate response
        response = generate_response(
            self.turn, user_message, self.preferences,
            recs, self.descriptions, self.client, self.history
        )

        self.history.append({"role": "assistant", "content": response})
        return response

    def get_recommendations(self) -> list:
        """Return the most recent recommendations."""
        return self.last_recommendations

    def reset(self):
        """Start a fresh conversation."""
        self.preferences = UserPreferences()
        self.history = [self.history[0]]  # keep system prompt
        self.turn = 0
        self.last_recommendations = []


# --------------------------------------------------------------------------
# 6. Rule-based fallback (no API key needed)
# --------------------------------------------------------------------------

class RuleBasedRecommender:
    """
    Simple rule-based conversational recommender.
    Works without any API key — demonstrates the concept offline.
    """

    BRAND_MAP = {
        "gaming": ["Steam", "PlayStation", "Xbox"],
        "music":  ["Spotify", "iTunes", "Apple Music"],
        "movies": ["Netflix", "Amazon Prime", "Disney+"],
        "food":   ["DoorDash", "Uber Eats", "Starbucks"],
        "shopping": ["Amazon", "Target", "Walmart"],
        "default": ["Amazon", "Apple", "Google Play"],
    }

    def __init__(self, descriptions: dict):
        self.descriptions = descriptions
        self.preferences  = UserPreferences()
        self.turn = 0

    def chat(self, user_message: str) -> str:
        self.turn += 1
        msg_lower = user_message.lower()

        # Simple keyword extraction
        for interest in ["gaming", "music", "movies", "food", "shopping"]:
            if interest in msg_lower:
                self.preferences.interests.append(interest)

        # Budget extraction
        import re
        amounts = re.findall(r'\$?(\d+)', msg_lower)
        if amounts:
            self.preferences.budget_max = float(max(amounts, key=int))

        # Occasion
        for occ in ["birthday", "holiday", "christmas", "graduation",
                    "wedding", "thank"]:
            if occ in msg_lower:
                self.preferences.occasion = occ
                break

        self.preferences.raw_query = user_message

        # Generate response
        if self.turn == 1 and not self.preferences.interests:
            return ("Hi! I can help you find the perfect gift card. "
                    "What are their interests — gaming, music, streaming, "
                    "food, or general shopping?")

        interests = self.preferences.interests or ["default"]
        brands = []
        for interest in interests:
            brands.extend(self.BRAND_MAP.get(interest, []))
        brands = brands[:3] or self.BRAND_MAP["default"]

        budget_str = (f"${self.preferences.budget_max:.0f}"
                      if self.preferences.budget_max < 999 else "any budget")
        occasion = (f"for {self.preferences.occasion}"
                    if self.preferences.occasion else "")

        return (f"Based on your preferences {occasion}, "
                f"I'd recommend: {', '.join(brands)} gift cards. "
                f"These are great for {', '.join(interests or ['general use'])} "
                f"and available at {budget_str}.")


# --------------------------------------------------------------------------
# 7. Sanity check — multi-turn demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)
    from semantic_retrieval import (generate_item_descriptions,
                                            tfidf_embeddings,
                                            SemanticRetriever)

    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, _ = train_test_split(df)

    all_items    = sorted(df["item_id"].unique().tolist())
    descriptions = generate_item_descriptions(all_items)
    item_ids, embeddings, vectorizer, pca = tfidf_embeddings(
        descriptions, n_components=64
    )
    retriever = SemanticRetriever(item_ids, embeddings, vectorizer, pca)

    api_key = os.environ.get("OPENAI_API_KEY", "")

    print("=" * 55)
    print("Conversational Recommender Demo")
    print("=" * 55)

    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        agent = ConversationalRecommender(retriever, descriptions, client)
        print("Mode: LLM-powered (gpt-4o-mini)\n")
    else:
        agent = RuleBasedRecommender(descriptions)
        print("Mode: Rule-based (no API key)\n")
        print("Set OPENAI_API_KEY for LLM-powered dialogue.\n")

    # Multi-turn demo conversation
    turns = [
        "I need a gift for my best friend's birthday",
        "He's really into gaming and streaming services",
        "Budget is around $50",
    ]

    for user_msg in turns:
        print(f"User : {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response}")
        print()

    # Show final recommendations
    if hasattr(agent, 'get_recommendations'):
        recs = agent.get_recommendations()
        if recs:
            print("Final recommendations extracted:")
            for item_id, desc, score in recs[:3]:
                print(f"  {desc[:60]}  (score={score:.3f})")