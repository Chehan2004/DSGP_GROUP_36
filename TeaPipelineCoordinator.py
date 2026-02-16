class TeaPipelineCoordinator:
    """
    Orchestrates a sequential multi-model pipeline for tea leaf health and quality analysis.
    
    This coordinator manages the lifecycle of 5 specialized engines, ensuring data 
    transformation and conditional logic (Gatekeeping) between localization and 
    final grading.
    """

    def __init__(self, detector, validator, diagnoser, expert, grader):
        """
        Args:
            detector: YOLO-based model for leaf localization and cropping.
            validator: Binary classifier to filter non-tea leaf artifacts.
            diagnoser: Multi-class classifier for disease identification.
            expert: Logic-based system for treatment mapping.
            grader: Feature extraction model for quality and maturity assessment.
        """
        self.detector = detector
        self.validator = validator
        self.diagnoser = diagnoser
        self.expert = expert
        self.grader = grader

    def run_analysis(self, image_path: str) -> list[dict]:
        """
        Executes the full diagnostic workflow on an input image.
        
        The process follows a 'Fail-Fast' logic: if the Validator fails, downstream 
        Disease and Grading models are bypassed to save compute resources.

        Args:
            image_path: System path to the raw high-resolution field image.

        Returns:
            A list of dictionaries, where each entry represents a validated tea leaf 
            with its associated health status, treatment, and quality metrics.
        """
        # Stage 1: Localize all potential leaf candidates
        crops, boxes = self.detector.extract_leaf_crops(image_path)
        final_results = []

        for i, crop in enumerate(crops):
            # Stage 2: Validation (The Gatekeeper)
            # Prevents 'False Positives' (background noise) from being diagnosed.
            if not self.validator.is_tea_leaf(crop):
                continue
            
            # Stage 3: Feature Extraction & Diagnosis
            disease = self.diagnoser.detect_disease(crop)
            treatment = self.expert.get_advice(disease)
            
            # Stage 4: Commercial Valuation
            metrics = self.grader.evaluate_quality(crop)
            
            # Stage 5: Data Aggregation (Final Report Construction)
            final_results.append({
                "id": i + 1,
                "bbox": boxes[i],
                "health_status": disease,
                "treatment_protocol": treatment,
                "maturity_index": metrics.maturity_level,
                "commercial_grade": metrics.quality_grade
            })

        return final_results