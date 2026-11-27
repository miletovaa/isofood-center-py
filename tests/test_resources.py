from unittest.mock import patch

import pytest

from isofood_center.client import OrgClient

from .helpers import FakeResponse


def _authed_client():
    return OrgClient("https://dma.example.org", token="preset-token")


def test_list_unwraps_bare_array():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}, {"id": 2}])):
        items = client.compounds.list().get()
    assert items == [{"id": 1}, {"id": 2}]


def test_list_unwraps_data_key():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, {"data": [{"id": 1}]})):
        items = client.compounds.list().get()
    assert items == [{"id": 1}]


def test_list_walks_every_page_when_all_pages_true():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 2}})
    page2 = FakeResponse(200, {"data": [{"id": 2}], "meta": {"current_page": 2, "last_page": 2}})
    with patch("requests.Session.request", side_effect=[page1, page2]):
        items = client.compounds.list(all_pages=True).get()
    assert items == [{"id": 1}, {"id": 2}]


def test_list_passes_filters_as_query_params():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [])) as mock_request:
        client.compounds.list(name="glucose").get()
    _, kwargs = mock_request.call_args
    assert kwargs["params"] == {"name": "glucose"}


def test_get_unwraps_single_resource():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, {"data": {"id": 1}})):
        item = client.compounds.get(1)
    assert item == {"id": 1}


def test_create_posts_fields_to_endpoint():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, {"id": 1})) as mock_request:
        client.compounds.create(canonical_name="Glucose")
    method, url = mock_request.call_args.args[:2]
    assert method == "POST"
    assert url == "https://dma.example.org/api/compounds"
    assert mock_request.call_args.kwargs["json"] == {"canonical_name": "Glucose"}


def test_update_puts_to_item_endpoint():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, {"id": 1})) as mock_request:
        client.compounds.update(1, canonical_name="Dextrose")
    method, url = mock_request.call_args.args[:2]
    assert method == "PUT"
    assert url == "https://dma.example.org/api/compounds/1"


def test_bulk_create_collects_errors_without_raising():
    client = _authed_client()
    responses = [
        FakeResponse(200, {"id": 1}),
        FakeResponse(422, {"errors": {"canonical_name": ["required"]}}),
    ]
    with patch("requests.Session.request", side_effect=responses):
        result = client.compounds.bulk_create([{"canonical_name": "Glucose"}, {"canonical_name": ""}])
    assert result["created"] == [{"id": 1}]
    assert len(result["errors"]) == 1


def test_nested_project_compounds_list_uses_parent_path():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [])) as mock_request:
        client.project_compounds.list(project_id=7).get()
    _, url = mock_request.call_args.args[:2]
    assert url == "https://dma.example.org/api/projects/7/compounds"


def test_nested_project_compounds_list_accepts_positional_id():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [])) as mock_request:
        client.project_compounds.list(7).get()
    _, url = mock_request.call_args.args[:2]
    assert url == "https://dma.example.org/api/projects/7/compounds"


def test_nested_experiment_records_list_uses_parent_path():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [])) as mock_request:
        client.experiment_records.list(experiment_id=3).get()
    _, url = mock_request.call_args.args[:2]
    assert url == "https://dma.example.org/api/experiments/3/records"


def test_nested_samples_list_uses_parent_path():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [])) as mock_request:
        client.samples.list(project_id=12).get()
    _, url = mock_request.call_args.args[:2]
    assert url == "https://dma.example.org/api/projects/12/samples"


def test_samples_update_uses_shallow_path():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, {"id": 6})) as mock_request:
        client.samples.update(6, sample_group="food")
    method, url = mock_request.call_args.args[:2]
    assert method == "PUT"
    assert url == "https://dma.example.org/api/samples/6"


def test_experiment_records_list_by_project_hits_project_scoped_endpoint():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [])) as mock_request:
        client.experiment_records.list_by_project(12, record_type="result_stable_isotopes").get()
    _, url = mock_request.call_args.args[:2]
    _, kwargs = mock_request.call_args
    assert url == "https://dma.example.org/api/projects/12/experiment-records"
    assert kwargs["params"] == {"record_type": "result_stable_isotopes"}


def test_experiment_records_list_by_project_defaults_to_all_pages():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 2}})
    page2 = FakeResponse(200, {"data": [{"id": 2}], "meta": {"current_page": 2, "last_page": 2}})
    with patch("requests.Session.request", side_effect=[page1, page2]):
        items = client.experiment_records.list_by_project(12, record_type="result_stable_isotopes").get()
    assert items.count() == 2


def test_nested_project_compounds_update_uses_shallow_path():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, {"id": 42})) as mock_request:
        client.project_compounds.update(42, is_mapped=True)
    method, url = mock_request.call_args.args[:2]
    assert method == "PUT"
    assert url == "https://dma.example.org/api/project-compounds/42"


def test_list_count_matches_length():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}, {"id": 2}, {"id": 3}])):
        count = client.compounds.list().count()
    assert count == 3


def test_list_names_prefers_name_over_canonical_name():
    client = _authed_client()
    payload = [{"name": "Some Project"}, {"canonical_name": "Glucose"}]
    with patch("requests.Session.request", return_value=FakeResponse(200, payload)):
        names = client.compounds.list().names()
    assert names == ["Some Project", "Glucose"]


def test_list_names_falls_back_through_custom_name_and_title():
    client = _authed_client()
    payload = [{"custom_name": "Peak 12"}, {"title": "Untitled"}]
    with patch("requests.Session.request", return_value=FakeResponse(200, payload)):
        names = client.project_compounds.list(project_id=1).names()
    assert names == ["Peak 12", "Untitled"]


def test_list_names_raises_when_no_name_like_field_present():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1, "mz": 180.0}])):
        with pytest.raises(KeyError):
            client.compounds.list().names()


def test_list_result_still_equals_a_plain_list():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}])):
        items = client.compounds.list()
        assert items == [{"id": 1}]
    assert isinstance(items.get(), list)


def test_list_page_param_is_sent_and_pages_is_populated():
    client = _authed_client()
    response = FakeResponse(200, {"data": [{"id": 11}], "meta": {"current_page": 2, "last_page": 5}})
    with patch("requests.Session.request", return_value=response) as mock_request:
        query = client.compounds.list(page=2)
        pages = query.pages
    _, kwargs = mock_request.call_args
    assert kwargs["params"] == {"page": 2}
    assert pages == 5


def test_list_pages_is_none_for_unpaginated_response():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}])):
        pages = client.compounds.list().pages
    assert pages is None


def test_nested_list_page_param_is_sent_and_pages_is_populated():
    client = _authed_client()
    response = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 3}})
    with patch("requests.Session.request", return_value=response) as mock_request:
        query = client.project_compounds.list(project_id=7, page=1)
        pages = query.pages
    _, kwargs = mock_request.call_args
    assert kwargs["params"] == {"page": 1}
    assert pages == 3


def test_filtered_top_level_list_walks_all_pages_by_default():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 8, "canonical_name": "L-Glutamic acid"}],
                                "meta": {"current_page": 1, "last_page": 2}})
    page2 = FakeResponse(200, {"data": [{"id": 9, "canonical_name": "Palmitic acid"}],
                                "meta": {"current_page": 2, "last_page": 2}})
    with patch("requests.Session.request", side_effect=[page1, page2]):
        items = client.compounds.list(name="acid").get()  # no explicit all_pages
    assert items.count() == 2
    assert items.pages == 2


def test_unfiltered_top_level_list_does_not_walk_pages_by_default():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 500}})
    # side_effect has only one response queued: a second HTTP call would raise StopIteration
    with patch("requests.Session.request", side_effect=[page1]):
        items = client.compounds.list().get()  # no filters, no explicit all_pages
    assert items.count() == 1
    assert items.pages == 500


def test_explicit_page_disables_auto_all_pages_even_when_filtered():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 8, "canonical_name": "L-Glutamic acid"}],
                                "meta": {"current_page": 1, "last_page": 3}})
    with patch("requests.Session.request", side_effect=[page1]):
        items = client.compounds.list(name="acid", page=1).get()
    assert items.count() == 1
    assert items.pages == 3


def test_explicit_all_pages_false_overrides_filtered_default():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 8, "canonical_name": "L-Glutamic acid"}],
                                "meta": {"current_page": 1, "last_page": 2}})
    with patch("requests.Session.request", side_effect=[page1]):
        items = client.compounds.list(name="acid", all_pages=False).get()
    assert items.count() == 1
    assert items.pages == 2


def test_nested_list_walks_all_pages_by_default_without_filters():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 2}})
    page2 = FakeResponse(200, {"data": [{"id": 2}], "meta": {"current_page": 2, "last_page": 2}})
    with patch("requests.Session.request", side_effect=[page1, page2]):
        items = client.project_compounds.list(project_id=7).get()  # no filters, no explicit all_pages
    assert items.count() == 2
    assert items.pages == 2


# -- laziness: the whole point ------------------------------------------------

def test_list_makes_no_request_until_used():
    client = _authed_client()
    with patch("requests.Session.request") as mock_request:
        client.compounds.list(name="glucose")  # constructing the query alone
        mock_request.assert_not_called()


def test_count_on_a_huge_filtered_result_makes_exactly_one_request():
    client = _authed_client()
    # 30,000+ matches spread across 1,500+ pages in real life - .count() must
    # never walk them; only ever the one response queued below is available,
    # so a second request would raise StopIteration and fail the test.
    huge_page = FakeResponse(200, {"data": [{"id": i} for i in range(20)],
                                    "meta": {"current_page": 1, "last_page": 1533, "total": 30645}})
    with patch("requests.Session.request", side_effect=[huge_page]) as mock_request:
        count = client.compounds.list(like="canonical_name,acid").count()
    assert count == 30645
    assert mock_request.call_count == 1


def test_pages_on_a_huge_filtered_result_makes_exactly_one_request():
    client = _authed_client()
    huge_page = FakeResponse(200, {"data": [{"id": i} for i in range(20)],
                                    "meta": {"current_page": 1, "last_page": 1533, "total": 30645}})
    with patch("requests.Session.request", side_effect=[huge_page]) as mock_request:
        pages = client.compounds.list(like="canonical_name,acid").pages
    assert pages == 1533
    assert mock_request.call_count == 1


def test_count_without_total_meta_falls_back_to_length_of_the_single_response():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}, {"id": 2}])) as mock_request:
        count = client.compounds.list().count()
    assert count == 2
    assert mock_request.call_count == 1


def test_count_and_pages_share_the_same_cheap_fetch_no_duplicate_request():
    client = _authed_client()
    response = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 7, "total": 7}})
    with patch("requests.Session.request", side_effect=[response]) as mock_request:
        query = client.compounds.list(name="glucose")
        assert query.count() == 7
        assert query.pages == 7  # reuses the cached cheap fetch, no second request
    assert mock_request.call_count == 1


def test_get_caches_result_no_refetch_on_second_call():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}])) as mock_request:
        query = client.compounds.list()
        first = query.get()
        second = query.get()
    assert first is second
    assert mock_request.call_count == 1


def test_count_after_get_reads_cached_materialized_result_not_a_new_request():
    client = _authed_client()
    page1 = FakeResponse(200, {"data": [{"id": 1}], "meta": {"current_page": 1, "last_page": 2}})
    page2 = FakeResponse(200, {"data": [{"id": 2}], "meta": {"current_page": 2, "last_page": 2}})
    with patch("requests.Session.request", side_effect=[page1, page2]) as mock_request:
        query = client.compounds.list(name="glucose")
        query.get()  # walks both pages
        count = query.count()  # should read len(cached), not issue a 3rd request
    assert count == 2
    assert mock_request.call_count == 2


def test_iterating_the_query_materializes_it():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}, {"id": 2}])):
        query = client.compounds.list()
        rows = [row for row in query]
    assert rows == [{"id": 1}, {"id": 2}]


def test_indexing_the_query_materializes_it():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}, {"id": 2}])):
        query = client.compounds.list()
        first = query[0]
    assert first == {"id": 1}


def test_len_of_the_query_materializes_it():
    client = _authed_client()
    with patch("requests.Session.request", return_value=FakeResponse(200, [{"id": 1}, {"id": 2}])):
        query = client.compounds.list()
        assert len(query) == 2
