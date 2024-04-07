INPUT_SCHEMA = {
    'word_level_transcript': {
        'type': list,
        'required': True,
    },
    "pexels_api_key": {
        "type": str,
        "required": True,
    },
    # n_vids_hint=6,
      #n_searches_per_broll=2,
    "n_vids_hint": {
        "type": int,
        "required": False,
    },
    "n_searches_per_broll": {
        "type": int,
        "required": False,
    },
}
