"""Build gold_corpus.jsonl -- the entity/relation-level answer key for
extraction_quality.py.

Not run automatically: this is a build script, like context-graph-eval's
build-corpus command. Its output (gold_corpus.jsonl) is what gets committed
and read by the eval; the corpus text and entity/relation annotations both
live here as plain, reviewable Python so a human can check each call before
it becomes ground truth.

    uv run --package unstructured2graph python evals/build_gold_corpus.py

Two provenances, kept visible on every record (mirrors context-graph-eval's
own Tier 1/Tier 2 split -- adopted vs authored, never blended silently):

- "longmemeval": real conversational turns, sampled by hand from
  ../../context-graph/eval/corpus/tier1-longmemeval.jsonl's `context` fields
  as they stood when this corpus was built and copied here verbatim as
  literal text -- deliberately NOT looked up by index against that file at
  build time, since it is itself under active revision on another branch
  (stratification/format changes) and turn order/count is not a stable key
  to index into. Entity-rich, but this natural chat corpus turned out to
  carry almost no explicit named-entity-to-named-entity relations (grepped
  the whole corpus for "works at/for|founder|founded|CEO|professor at|
  based in|headquartered" -- zero matches). So relation coverage from this
  source is limited to the couple of clear `located_in` statements that do
  occur ("Outer Banks in North Carolina", "Rocky Mountains in Colorado").
- "authored": written for this eval, specifically to exercise relation
  scoring (works_for / founded / located_in) that the adopted text does not
  naturally contain. Fictional names/companies throughout.

Entity types are the project's own default_ontology.yaml vocabulary, so gold
labels, LightRAG's entity_types_guidance, and GLiNER2Backend's schema are all
speaking the same vocabulary. Relation types (works_for, founded, located_in)
are this eval's own small, hand-picked set, passed to GLiNER2Backend via
Ontology.relation_types -- LightRAG has no relation_types concept, so
relation-level scoring is only ever meaningful for GLiNER2 (see README.md).

LLM-assisted-then-human-reviewed: entity/relation spans below were drafted
with model assistance against the source text and then checked by hand
against the ontology vocabulary before being committed. Treat this corpus as
a solid first draft, not an immutable ground truth -- spot-check it before
leaning on any single number it produces.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_PATH = SCRIPT_DIR / "gold_corpus.jsonl"

# Sampled from context-graph/eval/corpus/tier1-longmemeval.jsonl's `context`
# fields (main, pre-stratification-rework revision). (chunk_id, text,
# entities, relations). entities: list of (text, type). relations: list of
# (type, head_text, tail_text).
LONGMEMEVAL_RECORDS = [
    (
        "lme-00",
        "user: I'm planning a trip to Jordan and I'm really interested in visiting Petra. Can you tell me more "
        "about the history of the city and its significance? By the way, I just learned a lot about it in a "
        "lecture at the History Museum about ancient civilizations this month.",
        [("Jordan", "Location"), ("Petra", "Location"), ("History Museum", "Organization")],
        [],
    ),
    (
        "lme-01",
        "user: I'm looking for some information on dinosaurs. I attended a guided tour at the Natural History "
        "Museum yesterday with my dad and was really impressed by the fossil collection on display, especially "
        "the new T-Rex skeleton they acquired. Do you know more about the habitat and behavior of the T-Rex?",
        [("Natural History Museum", "Organization"), ("T-Rex", "Creature")],
        [],
    ),
    (
        "lme-02",
        "user: I'm looking for some information on modern art movements. I just got back from a guided tour at "
        "the Museum of Modern Art focused on 20th-century modern art movements, and it really sparked my "
        "interest. Can you tell me more about the key artists associated with Cubism?",
        [("Museum of Modern Art", "Organization"), ("Cubism", "Concept")],
        [],
    ),
    (
        "lme-03",
        "user: I need some help finding new throw pillows for my couch. I just got a new coffee table from West "
        "Elm about three weeks ago, and it's really made my living room feel modern, but my old pillows are "
        "looking worn out. Can you give me some recommendations on where to find some affordable and stylish "
        "ones?",
        [("West Elm", "Organization")],
        [],
    ),
    (
        "lme-04",
        "user: I'm looking for some healthy meal ideas for my busy weekdays. Do you have any recommendations? By "
        "the way, I've been relying on food delivery services a lot lately - I had Domino's Pizza three times "
        "last week!",
        [("Domino's Pizza", "Organization")],
        [],
    ),
    (
        "lme-05",
        "user: I'm looking for some healthy recipe ideas for lunch. Do you have any suggestions? By the way, "
        "I've been really busy lately and have been relying on food delivery services, like this new one I "
        "found called Fresh Fusion - they have some great pre-made meals.",
        [("Fresh Fusion", "Organization")],
        [],
    ),
    (
        "lme-06",
        "user: I'm planning my own wedding and I was wondering if you could give me some tips on how to choose "
        "the perfect venue. By the way, I just got back from a friend's wedding last weekend, and it was amazing "
        "- the bride, Jen, looked stunning in her bohemian-inspired dress, and her husband, Tom, was clearly "
        "smitten with her. It was at a rustic barn in the countryside, and it was so cozy and relaxed.",
        [("Jen", "Person"), ("Tom", "Person")],
        [],
    ),
    (
        "lme-07",
        "user: I've had good experiences with the local bike shop downtown where I bought my Bell Zephyr helmet "
        "for $120. They did a great job with the tune-up last time, and the mechanic was knowledgeable and "
        "friendly. I might just go back there for my next tune-up.",
        [("Bell Zephyr", "Artifact")],
        [],
    ),
    (
        "lme-08",
        "user: I'm planning a trip to Walmart this weekend and I'm looking for some deals on baby essentials. Do "
        "you have any info on their current sales or promotions on diapers? By the way, I used a Buy One Get "
        "One Free coupon on Luvs diapers at Walmart today, which was a great deal!",
        [("Walmart", "Organization")],
        [],
    ),
    (
        "lme-09",
        "user: I'm planning a shopping trip to Target this weekend and I'm wondering if you have any info on "
        "their current sales and promotions. By the way, I just redeemed $12 cashback for a $10 Amazon gift "
        "card from Ibotta today, so I'm feeling pretty good about my savings so far!",
        [("Target", "Organization"), ("Amazon", "Organization"), ("Ibotta", "Organization")],
        [],
    ),
    (
        "lme-10",
        "user: I'm trying to plan my grocery shopping trip for this week. Can you help me find any good deals or "
        "sales on diapers and formula at ShopRite? By the way, I signed up for their rewards program today, so "
        "I'm hoping to maximize my points and savings.",
        [("ShopRite", "Organization")],
        [],
    ),
    (
        "lme-11",
        "user: I'm having some issues with my nasal spray prescription from Dr. Patel. Can you help me with some "
        "tips on how to use it more effectively? By the way, I just got back from a follow-up appointment with "
        "my dermatologist, Dr. Lee, to get a biopsy on a suspicious mole on my back, and thankfully it was "
        "benign.",
        [("Dr. Patel", "Person"), ("Dr. Lee", "Person")],
        [],
    ),
    (
        "lme-12",
        "user: I've been feeling really exhausted lately and was wondering if you could help me find some tips "
        "on how to boost my energy levels. By the way, I recently had a UTI and was prescribed antibiotics by "
        "my primary care physician, Dr. Smith, so I'm not sure if that's still affecting me.",
        [("Dr. Smith", "Person")],
        [],
    ),
    (
        "lme-13",
        "user: I'm planning to write an essay about ancient civilizations and I was wondering if you could "
        "provide me with some information on the significance of Pharaoh Tutankhamun's golden mask. By the way, "
        'I saw it in person today at the Metropolitan Museum of Art\'s "Ancient Egyptian Artifacts" exhibition '
        "- it was breathtaking!",
        [("Tutankhamun", "Person"), ("Metropolitan Museum of Art", "Organization")],
        [],
    ),
    (
        "lme-14",
        "user: I'm planning to visit the Modern Art Museum again soon and I was wondering if you could recommend "
        "any upcoming exhibitions or events that I shouldn't miss. By the way, I attended their guided tour of "
        '"The Evolution of Abstract Expressionism" today, led by Dr. Patel, which was fantastic - her insights '
        "into Pollock and Rothko's works were incredibly enlightening.",
        [
            ("Modern Art Museum", "Organization"),
            ("Dr. Patel", "Person"),
            ("Pollock", "Person"),
            ("Rothko", "Person"),
        ],
        [],
    ),
    (
        "lme-15",
        "user: I've been thinking about getting a new backpack for my upcoming trip to the Eastern Sierra, and I "
        "was wondering if you could recommend some good options for a lightweight backpack that's suitable for "
        "3-4 day trips? By the way, I just got back from a day hike to Muir Woods National Monument with my "
        "family today, and it was amazing!",
        [("Eastern Sierra", "Location"), ("Muir Woods National Monument", "Location")],
        [],
    ),
    (
        "lme-16",
        "user: I'm looking for some recommendations on camping gear. I recently got back from a solo camping "
        "trip to Yosemite and realized I need to upgrade some of my equipment. By the way, I just got back from "
        "a road trip with friends to Big Sur and Monterey today, and it was amazing!",
        [("Yosemite", "Location"), ("Big Sur", "Location"), ("Monterey", "Location")],
        [],
    ),
    (
        "lme-17",
        "user: I'm also thinking about visiting my friend Rachel who recently moved to a new apartment in the "
        "city. Do you know what the weather is like in the city this time of year?",
        [("Rachel", "Person")],
        [],
    ),
    (
        "lme-18",
        "user: I'm planning a trip to the Rocky Mountains in Colorado and I was wondering if you could recommend "
        "some good hiking trails and camping spots in the area. By the way, I just got back from an amazing "
        "5-day camping trip to Yellowstone National Park last month, and I'm still buzzing from the experience.",
        [("Rocky Mountains", "Location"), ("Colorado", "Location"), ("Yellowstone National Park", "Location")],
        [("located_in", "Rocky Mountains", "Colorado")],
    ),
    (
        "lme-19",
        "user: I was thinking of getting Emily to audition for a role, I'll definitely encourage her to give it "
        "a shot. The play I attended was actually a production of The Glass Menagerie, have you heard of it?",
        [("Emily", "Person"), ("The Glass Menagerie", "Content")],
        [],
    ),
    (
        "lme-20",
        "user: I'm looking for some information on cancer research and the latest developments in the field. By "
        "the way, I attended a charity gala organized by the Cancer Research Foundation at a fancy hotel in "
        "downtown today, where the event raises over $100,000 for cancer research. Can you tell me what are "
        "some of the most promising areas of research currently being explored?",
        [("Cancer Research Foundation", "Organization")],
        [],
    ),
    (
        "lme-21",
        "user: I'm having some issues with high nitrite levels in my tank. I've been doing partial water "
        "changes, but I'm not sure if I'm doing it correctly. Can you walk me through the process? By the way, "
        "I've had some experience with aquariums - I have a 5-gallon tank with a solitary betta fish named "
        "Finley, which I got from my cousin.",
        [("Finley", "Creature")],
        [],
    ),
    (
        "lme-22",
        "user: Hi! I'm in the process of buying a house and I need help with some calculations. I recently put "
        "in an offer on a 3-bedroom townhouse in the Brookside neighborhood on February 25th, and after some "
        "negotiations, we agreed on a price of $340,000. Can you help me estimate my monthly mortgage payments?",
        [("Brookside", "Location")],
        [],
    ),
    (
        "lme-23",
        "user: I'm looking for a home warranty to protect my new place from unexpected repairs. Can you "
        "recommend some providers and their prices? By the way, I've been searching for a home for a while now, "
        "and I've seen some properties that just didn't fit my budget, like that one in Cedar Creek on February "
        "1st - it was way out of my league.",
        [("Cedar Creek", "Location")],
        [],
    ),
    (
        "lme-24",
        "user: I'm looking for some book recommendations. I started reading 'The Nightingale' by Kristin Hannah "
        "today and I'm really into historical fiction right now. Can you suggest some other books in the same "
        "genre that you think I'd enjoy?",
        [("The Nightingale", "Content"), ("Kristin Hannah", "Person")],
        [],
    ),
    (
        "lme-25",
        "user: I just finished listening to 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari today, "
        "and it got me thinking about the impact of technology on human evolution. Can you tell me more about "
        "the latest advancements in AI and its potential applications in various industries?",
        [("Sapiens: A Brief History of Humankind", "Content"), ("Yuval Noah Harari", "Person")],
        [],
    ),
    (
        "lme-26",
        "user: I'm looking for some book recommendations. I just finished listening to 'The Power' by Naomi "
        "Alderman today and it really made me think. I'm interested in exploring more books that challenge my "
        "perspectives. Do you have any suggestions?",
        [("The Power", "Content"), ("Naomi Alderman", "Person")],
        [],
    ),
    (
        "lme-27",
        "user: I'm actually buying a $325,000 house, and I got pre-approved for $350,000 from Wells Fargo. What "
        "would be the estimated closing costs for my situation?",
        [("Wells Fargo", "Organization")],
        [],
    ),
    (
        "lme-28",
        'user: I\'m looking for some recommendations for romantic comedies. I just saw "Coda" at the Seattle '
        "International Film Festival today, and I loved it. I attended SIFF for a week, watched 8 films, and "
        "even sat in on a panel discussion about film distribution and marketing. Anyway, what are some other "
        "rom-coms you'd suggest?",
        [("Coda", "Content"), ("Seattle International Film Festival", "Event")],
        [],
    ),
    (
        "lme-29",
        "user: I'm looking for some recommendations for independent films that explore themes of social justice "
        'and identity. I\'ve really been drawn to films like "Parasite" and "The Farewell" lately. By the '
        "way, I recently participated in the 48-hour film challenge at the Austin Film Festival, where my team "
        "and I had to write, shoot, and edit a short film within 48 hours - it was a wild ride!",
        [("Parasite", "Content"), ("The Farewell", "Content"), ("Austin Film Festival", "Event")],
        [],
    ),
    (
        "lme-30",
        "user: I'm trying to get organized and keep track of all the birthdays and milestones in my friends' and "
        "family's lives. Can you help me set up a calendar or reminder system to ensure I don't miss any "
        "special days? By the way, I just heard that my friend from college, David, had a baby boy named Jasper "
        "a few weeks ago - I'm still getting used to keeping track of all these new additions!",
        [("David", "Person"), ("Jasper", "Person")],
        [],
    ),
    (
        "lme-31",
        "user: I'm planning a baby gift for my aunt's twins, Ava and Lily, who were born in April. Can you give "
        "me some gift ideas for newborn twins?",
        [("Ava", "Person"), ("Lily", "Person")],
        [],
    ),
    (
        "lme-32",
        "user: I'm planning a get-together with some friends soon and I want to make sure I have all the latest "
        "updates on their kids. Do you think you could help me come up with a list of all the new babies and "
        "kids in our circle? Oh, by the way, I just remembered that our friends Mike and Emma welcomed their "
        "first baby, a girl named Charlotte, a few weeks after Rachel's baby shower.",
        [("Mike", "Person"), ("Emma", "Person"), ("Charlotte", "Person"), ("Rachel", "Person")],
        [],
    ),
    (
        "lme-33",
        "user: I'm planning another road trip and I'm thinking of going to a coastal town. Do you have any "
        "recommendations? By the way, I've had some great experiences with coastal trips, like my recent trip "
        "to Outer Banks in North Carolina - it only took me four hours to drive there from my place.",
        [("Outer Banks", "Location"), ("North Carolina", "Location")],
        [("located_in", "Outer Banks", "North Carolina")],
    ),
    (
        "lme-34",
        "user: I'm planning a new road trip and need some help with route planning. I've had some great "
        "experiences with my GPS device, like when I drove for six hours to Washington D.C. recently, but I'm "
        "not sure about the best route to take for my next trip. Can you suggest some options for me?",
        [("Washington D.C.", "Location")],
        [],
    ),
    (
        "lme-35",
        "user: I've used Trello in my previous role as a marketing specialist at a small startup and I'm "
        "familiar with its features. But I'm interested in exploring other options as well. Could you tell me "
        "more about ClickUp? How does it differ from Trello, and what are its strengths and weaknesses?",
        [("Trello", "Artifact"), ("ClickUp", "Artifact")],
        [],
    ),
]

# Written for this eval -- see module docstring. (chunk_id, text, entities, relations)
AUTHORED_RECORDS = [
    (
        "auth-01",
        "Alice Johnson works for Acme Corporation as a senior engineer on the graph database team.",
        [("Alice Johnson", "Person"), ("Acme Corporation", "Organization")],
        [("works_for", "Alice Johnson", "Acme Corporation")],
    ),
    (
        "auth-02",
        "Marco Santini founded Lumen Robotics in Turin in 2019 after leaving his previous job at Siemens.",
        [
            ("Marco Santini", "Person"),
            ("Lumen Robotics", "Organization"),
            ("Turin", "Location"),
            ("Siemens", "Organization"),
        ],
        [
            ("founded", "Marco Santini", "Lumen Robotics"),
            ("located_in", "Lumen Robotics", "Turin"),
            ("works_for", "Marco Santini", "Siemens"),
        ],
    ),
    (
        "auth-03",
        "Priya Raman leads the platform team at Nimbus Systems, which is headquartered in Austin, Texas.",
        [("Priya Raman", "Person"), ("Nimbus Systems", "Organization"), ("Austin", "Location")],
        [("works_for", "Priya Raman", "Nimbus Systems"), ("located_in", "Nimbus Systems", "Austin")],
    ),
    (
        "auth-04",
        "The Wren Foundation was founded by Clara Osei to support renewable-energy research across West Africa.",
        [("Clara Osei", "Person"), ("Wren Foundation", "Organization"), ("West Africa", "Location")],
        [("founded", "Clara Osei", "Wren Foundation")],
    ),
    (
        "auth-05",
        "David Kim works for Brightline Analytics, a data consultancy based in Seattle.",
        [("David Kim", "Person"), ("Brightline Analytics", "Organization"), ("Seattle", "Location")],
        [("works_for", "David Kim", "Brightline Analytics"), ("located_in", "Brightline Analytics", "Seattle")],
    ),
]


def _spans(chunk_id: str, text: str, entities: list[tuple[str, str]]) -> list[dict]:
    out = []
    for entity_text, entity_type in entities:
        start = text.find(entity_text)
        if start == -1:
            raise ValueError(f"{chunk_id}: entity text {entity_text!r} not found verbatim in chunk text: {text!r}")
        out.append({"text": entity_text, "type": entity_type, "start": start, "end": start + len(entity_text)})
    return out


def _relations(relations: list[tuple[str, str, str]]) -> list[dict]:
    return [{"type": rel_type, "head": head, "tail": tail} for rel_type, head, tail in relations]


def build() -> list[dict]:
    records = []
    for source, source_records in (("longmemeval", LONGMEMEVAL_RECORDS), ("authored", AUTHORED_RECORDS)):
        for chunk_id, text, entities, relations in source_records:
            records.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": text,
                    "entities": _spans(chunk_id, text, entities),
                    "relations": _relations(relations),
                }
            )
    return records


def main() -> None:
    records = build()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
