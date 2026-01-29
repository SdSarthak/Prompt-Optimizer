"""
Prompt Optimizer - AI-powered prompt improvement and refinement
Enhances rough prompts into optimized, effective instructions for LLMs
"""

import os
from typing import Dict, List, Optional
from datetime import datetime


class PromptOptimizer:
    """Main prompt optimization system"""
    
    def __init__(self):
        self.optimization_history = []
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load optimization templates"""
        return {
            'clarity': "Make this prompt clearer and more specific",
            'structure': "Add better structure and organization to this prompt",
            'context': "Add relevant context and examples to this prompt",
            'constraints': "Add appropriate constraints and guidelines",
            'complete': "Transform this into a comprehensive, optimized prompt"
        }
    
    def optimize(self, 
                rough_prompt: str,
                optimization_type: str = 'complete',
                target_model: str = 'general') -> Dict[str, any]:
        """
        Optimize a rough prompt
        
        Args:
            rough_prompt: Original prompt text
            optimization_type: Type of optimization to apply
            target_model: Target LLM model (e.g., 'gpt', 'claude', 'gemini')
            
        Returns:
            Dictionary with optimized prompt and metadata
        """
        print(f"Optimizing prompt for {target_model}...")
        
        # Analyze rough prompt
        analysis = self._analyze_prompt(rough_prompt)
        
        # Apply optimizations
        optimized = self._apply_optimizations(
            rough_prompt,
            analysis,
            optimization_type,
            target_model
        )
        
        # Record optimization
        result = {
            'original': rough_prompt,
            'optimized': optimized,
            'analysis': analysis,
            'optimization_type': optimization_type,
            'target_model': target_model,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(result)
        
        return result
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, any]:
        """Analyze prompt quality and characteristics"""
        words = prompt.split()
        sentences = prompt.split('.')
        
        analysis = {
            'length': len(prompt),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'has_context': any(word in prompt.lower() for word in ['context', 'background', 'scenario']),
            'has_constraints': any(word in prompt.lower() for word in ['must', 'should', 'require', 'limit']),
            'has_examples': any(word in prompt.lower() for word in ['example', 'like', 'such as']),
            'has_format': any(word in prompt.lower() for word in ['format', 'structure', 'json', 'list']),
            'clarity_score': self._calculate_clarity_score(prompt)
        }
        
        return analysis
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate prompt clarity score (0-1)"""
        score = 0.5  # Base score
        
        # Increase score for specific elements
        if len(prompt) > 50:
            score += 0.1
        if any(word in prompt.lower() for word in ['please', 'help', 'create', 'generate']):
            score += 0.1
        if '?' in prompt or '.' in prompt:
            score += 0.1
        if any(word in prompt.lower() for word in ['detailed', 'specific', 'comprehensive']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _apply_optimizations(self, 
                           prompt: str,
                           analysis: Dict,
                           optimization_type: str,
                           target_model: str) -> str:
        """Apply optimizations to prompt"""
        
        optimized_parts = []
        
        # Add role/context if missing
        if not analysis['has_context']:
            optimized_parts.append(
                "You are an expert AI assistant. Your task is to provide detailed, "
                "accurate, and helpful responses."
            )
        
        # Add the core request with improvements
        optimized_parts.append(f"\n{prompt}")
        
        # Add structure requirements
        if not analysis['has_format']:
            optimized_parts.append(
                "\n\nPlease structure your response in a clear, organized manner."
            )
        
        # Add constraints if needed
        if not analysis['has_constraints'] and optimization_type == 'complete':
            optimized_parts.append(
                "\n\nRequirements:"
                "\n- Be specific and detailed"
                "\n- Use clear, professional language"
                "\n- Provide examples when relevant"
                "\n- Keep the response focused and concise"
            )
        
        # Add model-specific optimizations
        if target_model.lower() in ['gpt', 'openai']:
            optimized_parts.append(
                "\n\nNote: Provide a comprehensive response that balances "
                "detail with clarity."
            )
        elif target_model.lower() in ['claude', 'anthropic']:
            optimized_parts.append(
                "\n\nNote: Please think step-by-step and explain your reasoning."
            )
        elif target_model.lower() in ['gemini', 'google']:
            optimized_parts.append(
                "\n\nNote: Provide accurate information with relevant context."
            )
        
        return "".join(optimized_parts)
    
    def save_optimization(self, result: Dict, output_file: str = None):
        """Save optimization result to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"optimized_prompt_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PROMPT OPTIMIZATION RESULT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("ORIGINAL PROMPT:\n")
            f.write("-" * 70 + "\n")
            f.write(result['original'] + "\n\n")
            
            f.write("OPTIMIZED PROMPT:\n")
            f.write("-" * 70 + "\n")
            f.write(result['optimized'] + "\n\n")
            
            f.write("ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for key, value in result['analysis'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Saved optimization to {output_file}")
        return output_file
    
    def compare_prompts(self, prompt1: str, prompt2: str):
        """Compare two prompts"""
        analysis1 = self._analyze_prompt(prompt1)
        analysis2 = self._analyze_prompt(prompt2)
        
        print("\n" + "=" * 70)
        print("PROMPT COMPARISON")
        print("=" * 70)
        
        print("\nPrompt 1 Analysis:")
        for key, value in analysis1.items():
            print(f"  {key}: {value}")
        
        print("\nPrompt 2 Analysis:")
        for key, value in analysis2.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)


def main():
    """Main execution function"""
    print("=" * 70)
    print("Prompt Optimizer - AI-Powered Prompt Enhancement")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = PromptOptimizer()
    
    # Sample rough prompts
    rough_prompts = [
        "write a story",
        "explain machine learning",
        "help me code a website",
        "what is climate change"
    ]
    
    print("\nOptimizing sample prompts...")
    print("=" * 70)
    
    results = []
    for i, rough_prompt in enumerate(rough_prompts, 1):
        print(f"\n[{i}/{len(rough_prompts)}] Processing: '{rough_prompt}'")
        
        result = optimizer.optimize(
            rough_prompt,
            optimization_type='complete',
            target_model='general'
        )
        
        results.append(result)
        
        print(f"\nOriginal ({len(result['original'])} chars):")
        print(f"  {result['original']}")
        
        print(f"\nOptimized ({len(result['optimized'])} chars):")
        print(f"  {result['optimized'][:200]}...")
        
        print(f"\nClarity Score: {result['analysis']['clarity_score']:.2f}")
        
        # Save optimization
        output_file = f"optimized_prompt_{i}.txt"
        optimizer.save_optimization(result, output_file)
    
    # Compare first two prompts
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("Comparing first two optimizations...")
        optimizer.compare_prompts(
            results[0]['optimized'],
            results[1]['optimized']
        )
    
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print(f"Processed {len(results)} prompts")
    print("=" * 70)


if __name__ == "__main__":
    main()
