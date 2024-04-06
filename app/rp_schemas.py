INPUT_SCHEMA = {
    'word_level_transcript': {
        'type': list,
        'required': True,
    },
    'context_start_s' :
    {
        'type': int,
        'required': True
    },
    # context_end_s = 30
    # context_buffer_s = 5
    'context_end_s' : {
        'type': int,
        'required': True
    },
    'context_buffer_s' : {
        'type': int,
        'required': False,
    },
    
    #n_seach_phrases=3,
    # n_searches_per_phrase=5,
    # top_K=5,
    'n_search_phrases' : {
        'type': int,
        'required': False,
    },
    'n_searche_results_per_phrase' : {
        'type': int,
        'required': False,
    },
    'top_K' : {
        'type': int,
        'required': False,
    }
}
