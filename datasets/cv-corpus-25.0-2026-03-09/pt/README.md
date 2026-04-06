# *Português* &mdash; Portuguese (`pt`)

This datasheet is for cv-corpus-25.0-2026-03-09 of the Mozilla Common Voice *Scripted Speech* dataset for Portuguese [Português - `pt`]. The dataset contains 196698 clips representing 228.79 hours of recorded speech (187.33 hours validated) from 3817 speakers, recorded from a text corpus of 43,721 sentences.

## Language

### Variants

| Code | Variant | Clips | Speakers |
|---|---|---|---|
| pt-BR | Portuguese (Brasil) | 85,061 (43.2%) | 601 (15.7%) |
| pt-PT | Portuguese (Portugal) | 3,407 (1.7%) | 72 (1.9%) |

### Accents

| Code | Accent | Clips | Speakers |
|---|---|---|---|
| - |  | 87,116 (44.3%) | 332 (8.7%) |

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | 132,776 (67.5%) | 944 (24.7%) |
| female_feminine | Female, feminine | 11,409 (5.8%) | 135 (3.5%) |
| transgender | Transgender | - | - |
| non-binary | Non-binary | 10 (0.0%) | 1 (0.0%) |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 52,503 (26.7%) | 3,030 (79.4%) |

*Gender declared: 144,195 of 196,698 clips (73.3%), 787 of 3,817 speakers (20.6%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | 4,513 (2.3%) | 101 (2.6%) |
| twenties | Twenties | 70,498 (35.8%) | 493 (12.9%) |
| thirties | Thirties | 34,584 (17.6%) | 313 (8.2%) |
| fourties | Fourties | 25,440 (12.9%) | 158 (4.1%) |
| fifties | Fifties | 5,070 (2.6%) | 71 (1.9%) |
| sixties | Sixties | 8,775 (4.5%) | 19 (0.5%) |
| seventies | Seventies | 16 (0.0%) | 2 (0.1%) |
| eighties | Eighties | - | - |
| nineties | Nineties | - | - |
| - | Unspecified | 47,802 (24.3%) | 2,978 (78.0%) |

*Age declared: 148,896 of 196,698 clips (75.7%), 839 of 3,817 speakers (22.0%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 161,047 (81.9%) |
| Invalidated | 7,872 (4.0%) |
| Other | 27,779 (14.1%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 23,092 (14.3%) |
| Dev | 9,669 (6.0%) |
| Test | 9,670 (6.0%) |

*Training split coverage: 42,431 of 161,047 validated clips (26.3%)*

The dataset contains 161047 validated, 7872 invalidated, and 27779 unresolved clips. The average clip duration is 4.188 seconds.

## Text corpus

**Validated sentences:** 43,613

| Category | Count |
|---|---|
| Unvalidated sentences | 108 |
| Pending sentences | 8 |
| Rejected sentences | 100 |
| Reported sentences | 2,839 |

The corpus contains 43,721 sentences: 43,613 validated and 108 unvalidated (8 pending review, 100 rejected), with 2,839 reported for review.

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *Itobi*
2. *Nós viemos de La Pobla de Segur.*
3. *O que o amor não vê, ele acredita.*
4. *Podemos por favor sair agora?*
5. *Leo Pericles, Sofia Manzano*

### Sources

| Source | Sentences |
|---|---|
| sentence-collector | 41,016 (94.0%) |
| Autocitação | 1,311 (3.0%) |
| Other | 1,286 (2.9%) |

### Text domains

| Code | Domain | Clips | Speakers |
|---|---|---|---|
| general | General | 1,426 (0.7%) | 280 (7.3%) |
| agriculture_food | Agriculture and Food | 198 (0.1%) | 72 (1.9%) |
| automotive_transport | Automotive and Transport | 37 (0.0%) | 23 (0.6%) |
| finance | Finance | 27 (0.0%) | 18 (0.5%) |
| service_retail | Service and Retail | 25 (0.0%) | 20 (0.5%) |
| healthcare | Healthcare | 206 (0.1%) | 92 (2.4%) |
| history_law_government | History, Law and Government | 67 (0.0%) | 40 (1.0%) |
| media_entertainment | Media and Entertainment | 82 (0.0%) | 54 (1.4%) |
| nature_environment | Nature and Environment | 53 (0.0%) | 32 (0.8%) |
| news_current_affairs | News and Current Affairs | 4 (0.0%) | 4 (0.1%) |
| technology_robotics | Technology and Robotics | 193 (0.1%) | 84 (2.2%) |
| language_fundamentals | Language Fundamentals | 17 (0.0%) | 13 (0.3%) |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `text` - supposed transcription of the audio
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here
- `prompt_upvotes` - number of upvotes the sentence prompt received
- `prompt_reports` - number of reports the sentence prompt received
- `is_edited` - whether the clip's transcription has been edited

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/pt/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Speak](https://commonvoice.mozilla.org/pt/speak)
- [Write](https://commonvoice.mozilla.org/pt/write)
- [Listen](https://commonvoice.mozilla.org/pt/listen)
- [Review](https://commonvoice.mozilla.org/pt/review)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
