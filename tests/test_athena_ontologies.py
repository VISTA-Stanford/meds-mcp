"""
Unit tests for AthenaOntology and LazyAthenaOntology classes.
"""

import pytest
import tempfile
import os
import zipfile
import pandas as pd
import polars as pl
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

from meds_mcp.server.tools.ontologies import (
    AthenaOntology, 
    LazyAthenaOntology, 
    AthenaFileReader,
    create_lazy_ontology
)


class TestAthenaFileReader:
    """Test the AthenaFileReader utility class."""
    
    def test_file_reader_with_directory(self, tmp_path):
        """Test reading from a directory structure."""
        # Create test CSV files
        concept_data = "concept_id\tvocabulary_id\tconcept_code\tconcept_name\tinvalid_reason\tstandard_concept\n"
        concept_data += "1\tSNOMED\t123456\tTest Concept\t\tS\n"
        
        relationship_data = "concept_id_1\tconcept_id_2\trelationship_id\n"
        relationship_data += "1\t2\tMaps to\n"
        
        ancestor_data = "ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\n"
        ancestor_data += "2\t1\t1\n"
        
        # Write test files
        (tmp_path / "CONCEPT.csv").write_text(concept_data)
        (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_data)
        (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_data)
        
        with AthenaFileReader(str(tmp_path)) as reader:
            assert not reader.is_zip
            concepts_df = reader.read_csv("CONCEPT.csv").collect()
            assert concepts_df.height == 1
            assert concepts_df.row(0)[0] == "1"  # concept_id
    
    def test_file_reader_with_zip(self, tmp_path):
        """Test reading from a ZIP archive."""
        # Create test data
        concept_data = "concept_id\tvocabulary_id\tconcept_code\tconcept_name\tinvalid_reason\tstandard_concept\n"
        concept_data += "1\tSNOMED\t123456\tTest Concept\t\tS\n"
        
        relationship_data = "concept_id_1\tconcept_id_2\trelationship_id\n"
        relationship_data += "1\t2\tMaps to\n"
        
        ancestor_data = "ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\n"
        ancestor_data += "2\t1\t1\n"
        
        # Create ZIP file
        zip_path = tmp_path / "athena.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("CONCEPT.csv", concept_data)
            zf.writestr("CONCEPT_RELATIONSHIP.csv", relationship_data)
            zf.writestr("CONCEPT_ANCESTOR.csv", ancestor_data)
        
        with AthenaFileReader(str(zip_path)) as reader:
            assert reader.is_zip
            concepts_df = reader.read_csv("CONCEPT.csv").collect()
            assert concepts_df.height == 1
            assert concepts_df.row(0)[0] == "1"  # concept_id


class TestLazyAthenaOntology:
    """Test the LazyAthenaOntology class."""
    
    @pytest.fixture
    def sample_athena_data(self, tmp_path):
        """Create sample Athena data for testing."""
        # Create sample CONCEPT.csv
        concept_data = [
            ["concept_id", "vocabulary_id", "concept_code", "concept_name", "invalid_reason", "standard_concept"],
            ["1", "SNOMED", "123456", "Clinical Finding", "", "S"],
            ["2", "SNOMED", "789012", "Procedure", "", "S"],
            ["3", "RxNorm", "ABC123", "Medication", "", "S"],
            ["4", "SNOMED", "999999", "Invalid Concept", "D", ""],  # Invalid concept
            ["5", "ICD10CM", "E11.9", "Diabetes Type 2", "", ""],  # Non-standard
        ]
        
        concept_content = "\n".join(["\t".join(row) for row in concept_data])
        (tmp_path / "CONCEPT.csv").write_text(concept_content)
        
        # Create sample CONCEPT_RELATIONSHIP.csv
        relationship_data = [
            ["concept_id_1", "concept_id_2", "relationship_id"],
            ["5", "1", "Maps to"],  # Non-standard maps to standard
            ["1", "2", "Is a"],  # Not "Maps to" - should be filtered
        ]
        
        relationship_content = "\n".join(["\t".join(row) for row in relationship_data])
        (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_content)
        
        # Create sample CONCEPT_ANCESTOR.csv
        ancestor_data = [
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
            ["1", "2", "1"],  # Direct parent-child
            ["1", "3", "2"],  # Grandparent - should be filtered
        ]
        
        ancestor_content = "\n".join(["\t".join(row) for row in ancestor_data])
        (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_content)
        
        return tmp_path
    
    def test_lazy_ontology_loading(self, sample_athena_data):
        """Test loading LazyAthenaOntology from sample data."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(
            str(sample_athena_data),
            ignore_invalid=True
        )
        
        # Should have 4 concepts (excluding invalid one)
        assert len(ontology) >= 4
        
        # Check mapping dictionaries are populated
        assert len(ontology.concept_id_to_code_map) >= 4
        assert len(ontology.code_to_concept_id_map) >= 4
        
        # Check specific mappings
        assert "SNOMED/123456" in ontology.code_to_concept_id_map
        assert "SNOMED/789012" in ontology.code_to_concept_id_map
        assert "RxNorm/ABC123" in ontology.code_to_concept_id_map
        
        # Invalid concept should be excluded when ignore_invalid=True
        assert "SNOMED/999999" not in ontology.code_to_concept_id_map
    
    def test_lazy_ontology_loading_with_invalid(self, sample_athena_data):
        """Test loading LazyAthenaOntology including invalid concepts."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(
            str(sample_athena_data),
            ignore_invalid=False
        )
        
        # Should include invalid concept when ignore_invalid=False
        assert "SNOMED/999999" in ontology.code_to_concept_id_map
    
    def test_get_description(self, sample_athena_data):
        """Test getting descriptions for codes."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test existing codes
        assert ontology.get_description("SNOMED/123456") == "Clinical Finding"
        assert ontology.get_description("SNOMED/789012") == "Procedure"
        assert ontology.get_description("RxNorm/ABC123") == "Medication"
        
        # Test non-existing code
        assert ontology.get_description("NONEXISTENT/000000") is None
    
    def test_get_parents(self, sample_athena_data):
        """Test getting parent codes."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test concept with parents
        parents = ontology.get_parents("SNOMED/789012")
        assert "SNOMED/123456" in parents
        
        # Test concept with Maps to relationship
        parents = ontology.get_parents("ICD10CM/E11.9")
        assert "SNOMED/123456" in parents  # Maps to relationship
        
        # Test concept without parents
        parents = ontology.get_parents("SNOMED/123456")
        assert len(parents) == 0
        
        # Test non-existing code
        parents = ontology.get_parents("NONEXISTENT/000000")
        assert len(parents) == 0
    
    def test_get_children(self, sample_athena_data):
        """Test getting child codes."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test concept with children
        children = ontology.get_children("SNOMED/123456")
        assert "SNOMED/789012" in children
        
        # Test concept without children
        children = ontology.get_children("SNOMED/789012")
        assert len(children) == 0
        
        # Test non-existing code
        children = ontology.get_children("NONEXISTENT/000000")
        assert len(children) == 0
    
    def test_get_code_metadata(self, sample_athena_data):
        """Test getting code metadata."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test existing code
        metadata = ontology.get_code_metadata("SNOMED/123456")
        expected = {
            "code": "SNOMED/123456",
            "description": "Clinical Finding",
            "vocabulary": "SNOMED"
        }
        assert metadata == expected
        
        # Test non-existing code
        metadata = ontology.get_code_metadata("NONEXISTENT/000000")
        assert "error" in metadata
    
    def test_get_ancestor_subgraph(self, sample_athena_data):
        """Test generating ancestor subgraphs."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test subgraph generation
        G = ontology.get_ancestor_subgraph("SNOMED/789012")
        assert isinstance(G, nx.DiGraph)
        
        # Should include the starting node and its parent
        assert "SNOMED/789012" in G.nodes()
        assert "SNOMED/123456" in G.nodes()
        
        # Check edge direction (child -> parent)
        assert G.has_edge("SNOMED/789012", "SNOMED/123456")
        
        # Test vocabulary filtering
        G_filtered = ontology.get_ancestor_subgraph("SNOMED/789012", vocabularies=["SNOMED"])
        assert "SNOMED/789012" in G_filtered.nodes()
        assert "SNOMED/123456" in G_filtered.nodes()
        
        # Test with non-matching vocabulary filter
        G_empty = ontology.get_ancestor_subgraph("SNOMED/789012", vocabularies=["RxNorm"])
        assert "SNOMED/789012" in G_empty.nodes()  # Starting node always included
        assert "SNOMED/123456" not in G_empty.nodes()  # Parent filtered out
    
    def test_get_descendant_subgraph(self, sample_athena_data):
        """Test generating descendant subgraphs."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test subgraph generation
        G = ontology.get_descendant_subgraph("SNOMED/123456")
        assert isinstance(G, nx.DiGraph)
        
        # Should include the starting node and its child
        assert "SNOMED/123456" in G.nodes()
        assert "SNOMED/789012" in G.nodes()
        
        # Check edge direction (child -> parent)
        assert G.has_edge("SNOMED/789012", "SNOMED/123456")
    
    def test_get_graph_metadata(self, sample_athena_data):
        """Test getting metadata for graph nodes."""
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        G = ontology.get_ancestor_subgraph("SNOMED/789012")
        metadata = ontology.get_graph_metadata(G)
        
        assert isinstance(metadata, dict)
        assert "SNOMED/789012" in metadata
        assert "SNOMED/123456" in metadata
        
        # Check metadata structure
        node_metadata = metadata["SNOMED/123456"]
        assert node_metadata["code"] == "SNOMED/123456"
        assert node_metadata["description"] == "Clinical Finding"
        assert node_metadata["vocabulary"] == "SNOMED"
    
    def test_code_metadata_parameter(self, sample_athena_data):
        """Test loading with custom code metadata."""
        custom_metadata = {
            "CUSTOM/001": {
                "description": "Custom Code",
                "parent_codes": ["SNOMED/123456"]
            }
        }
        
        ontology = LazyAthenaOntology.load_from_athena_snapshot(
            str(sample_athena_data),
            code_metadata=custom_metadata
        )
        
        # Custom metadata should be accessible
        assert ontology.get_description("CUSTOM/001") == "Custom Code"
        parents = ontology.get_parents("CUSTOM/001")
        assert "SNOMED/123456" in parents


class TestCreateLazyOntology:
    """Test the helper function for creating lazy ontologies."""
    
    def test_create_lazy_ontology_function(self, tmp_path):
        """Test the create_lazy_ontology helper function."""
        # Create minimal test data
        concept_data = "concept_id\tvocabulary_id\tconcept_code\tconcept_name\tinvalid_reason\tstandard_concept\n"
        concept_data += "1\tSNOMED\t123456\tTest Concept\t\tS\n"
        
        relationship_data = "concept_id_1\tconcept_id_2\trelationship_id\n"
        relationship_data += "1\t2\tMaps to\n"
        
        ancestor_data = "ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\n"
        ancestor_data += "2\t1\t1\n"
        
        (tmp_path / "CONCEPT.csv").write_text(concept_data)
        (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_data)
        (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_data)
        
        # Test helper function
        ontology = create_lazy_ontology(str(tmp_path))
        assert isinstance(ontology, LazyAthenaOntology)
        assert len(ontology) >= 1


class TestOntologyComparison:
    """Compare LazyAthenaOntology with regular AthenaOntology behavior."""
    
    @pytest.fixture
    def sample_athena_data(self, tmp_path):
        """Create sample Athena data for testing."""
        # Create sample CONCEPT.csv
        concept_data = [
            ["concept_id", "vocabulary_id", "concept_code", "concept_name", "invalid_reason", "standard_concept"],
            ["1", "SNOMED", "123456", "Clinical Finding", "", "S"],
            ["2", "SNOMED", "789012", "Procedure", "", "S"],
            ["3", "RxNorm", "ABC123", "Medication", "", "S"],
            ["4", "SNOMED", "999999", "Invalid Concept", "D", ""],  # Invalid concept
            ["5", "ICD10CM", "E11.9", "Diabetes Type 2", "", ""],  # Non-standard
        ]
        
        concept_content = "\n".join(["\t".join(row) for row in concept_data])
        (tmp_path / "CONCEPT.csv").write_text(concept_content)
        
        # Create sample CONCEPT_RELATIONSHIP.csv
        relationship_data = [
            ["concept_id_1", "concept_id_2", "relationship_id"],
            ["5", "1", "Maps to"],  # Non-standard maps to standard
            ["1", "2", "Is a"],  # Not "Maps to" - should be filtered
        ]
        
        relationship_content = "\n".join(["\t".join(row) for row in relationship_data])
        (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_content)
        
        # Create sample CONCEPT_ANCESTOR.csv
        ancestor_data = [
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
            ["1", "2", "1"],  # Direct parent-child
            ["1", "3", "2"],  # Grandparent - should be filtered
        ]
        
        ancestor_content = "\n".join(["\t".join(row) for row in ancestor_data])
        (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_content)
        
        return tmp_path
    
    @pytest.fixture
    def mock_regular_ontology(self):
        """Create a mock regular ontology for comparison."""
        description_map = {
            "SNOMED/123456": "Clinical Finding",
            "SNOMED/789012": "Procedure",
            "RxNorm/ABC123": "Medication"
        }
        
        parents_map = {
            "SNOMED/789012": {"SNOMED/123456"},
            "SNOMED/123456": set(),
            "RxNorm/ABC123": set()
        }
        
        return AthenaOntology(description_map, parents_map)
    
    def test_api_compatibility(self, sample_athena_data, mock_regular_ontology):
        """Test that LazyAthenaOntology has the same API as regular AthenaOntology."""
        lazy_ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Both should have the same methods
        lazy_methods = set(dir(lazy_ontology))
        regular_methods = set(dir(mock_regular_ontology))
        
        core_methods = {
            'get_code_metadata', 'get_description', 'get_parents', 'get_children',
            'get_ancestor_subgraph', 'get_descendant_subgraph', 'get_graph_metadata',
            '__len__'
        }
        
        assert core_methods.issubset(lazy_methods)
        assert core_methods.issubset(regular_methods)
    
    def test_len_behavior(self, sample_athena_data, mock_regular_ontology):
        """Test that __len__ works for both ontology types."""
        lazy_ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Both should return integers
        assert isinstance(len(lazy_ontology), int)
        assert isinstance(len(mock_regular_ontology), int)
        assert len(lazy_ontology) > 0
        assert len(mock_regular_ontology) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_files(self, tmp_path):
        """Test handling of missing required files."""
        # Create directory with missing files
        with pytest.raises((FileNotFoundError, RuntimeError)):
            LazyAthenaOntology.load_from_athena_snapshot(str(tmp_path))
    
    def test_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            LazyAthenaOntology.load_from_athena_snapshot("/nonexistent/path")
    
    def test_empty_dataframes(self, tmp_path):
        """Test handling of empty dataframes."""
        # Create empty CSV files with headers only
        concept_data = "concept_id\tvocabulary_id\tconcept_code\tconcept_name\tinvalid_reason\tstandard_concept\n"
        relationship_data = "concept_id_1\tconcept_id_2\trelationship_id\n"
        ancestor_data = "ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\n"
        
        (tmp_path / "CONCEPT.csv").write_text(concept_data)
        (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_data)
        (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_data)
        
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(tmp_path))
        assert len(ontology) == 0


class TestPerformance:
    """Performance and memory tests (basic checks)."""
    
    @pytest.fixture
    def sample_athena_data(self, tmp_path):
        """Create sample Athena data for testing."""
        # Create sample CONCEPT.csv
        concept_data = [
            ["concept_id", "vocabulary_id", "concept_code", "concept_name", "invalid_reason", "standard_concept"],
            ["1", "SNOMED", "123456", "Clinical Finding", "", "S"],
            ["2", "SNOMED", "789012", "Procedure", "", "S"],
            ["3", "RxNorm", "ABC123", "Medication", "", "S"],
            ["4", "SNOMED", "999999", "Invalid Concept", "D", ""],  # Invalid concept
            ["5", "ICD10CM", "E11.9", "Diabetes Type 2", "", ""],  # Non-standard
        ]
        
        concept_content = "\n".join(["\t".join(row) for row in concept_data])
        (tmp_path / "CONCEPT.csv").write_text(concept_content)
        
        # Create sample CONCEPT_RELATIONSHIP.csv
        relationship_data = [
            ["concept_id_1", "concept_id_2", "relationship_id"],
            ["5", "1", "Maps to"],  # Non-standard maps to standard
            ["1", "2", "Is a"],  # Not "Maps to" - should be filtered
        ]
        
        relationship_content = "\n".join(["\t".join(row) for row in relationship_data])
        (tmp_path / "CONCEPT_RELATIONSHIP.csv").write_text(relationship_content)
        
        # Create sample CONCEPT_ANCESTOR.csv
        ancestor_data = [
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
            ["1", "2", "1"],  # Direct parent-child
            ["1", "3", "2"],  # Grandparent - should be filtered
        ]
        
        ancestor_content = "\n".join(["\t".join(row) for row in ancestor_data])
        (tmp_path / "CONCEPT_ANCESTOR.csv").write_text(ancestor_content)
        
        return tmp_path
    
    def test_lazy_loading_performance(self, sample_athena_data):
        """Test that lazy loading is reasonably fast."""
        import time
        
        start_time = time.time()
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        load_time = time.time() - start_time
        
        # Should load reasonably quickly (adjust threshold as needed)
        assert load_time < 10.0  # 10 seconds max for test data
        assert len(ontology) > 0
    
    def test_query_performance(self, sample_athena_data):
        """Test that queries execute in reasonable time."""
        import time
        
        ontology = LazyAthenaOntology.load_from_athena_snapshot(str(sample_athena_data))
        
        # Test multiple query types
        start_time = time.time()
        
        for _ in range(10):  # Run multiple times
            _ = ontology.get_description("SNOMED/123456")
            _ = ontology.get_parents("SNOMED/789012")
            _ = ontology.get_children("SNOMED/123456")
        
        total_time = time.time() - start_time
        avg_time = total_time / 30  # 30 total operations
        
        # Each operation should be reasonably fast
        assert avg_time < 0.1  # 100ms average per operation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
