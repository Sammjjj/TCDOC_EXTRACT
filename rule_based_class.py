import pandas as pd
import re

category_to_subclusters = {
    'Apprenticeships, Internships, Placements': {
        'Apprentice & Trainee Schemes': ['apprenticeship', 'apprenticeships', 'trainee', 'scheme', 'schemes'],
        'Internships & Placements': ['internship', 'internships', 'placement', 'placements', 'work', 'experience', 'student'],
        'Development of Opportunities': ['develop', 'opportunities', 'offer', 'explore', 'create']
    },
    'Career Frameworks & Role Definition': {
        'Role Profiles & Job Families': ['role', 'roles', 'profiles', 'job', 'families', 'define', 'definition'],
        'Career Framework & Structure': ['framework', 'structure', 'map', 'mapping', 'pathways'],
        'Review & Alignment': ['review', 'align', 'grade', 'grading', 'benchmark']
    },
    'Career Pathways & Progression': {
        'Promotion & Progression Routes': ['promotion', 'progression', 'pathway', 'pathways', 'advancement'],
        'Career Development & Opportunities': ['career', 'development', 'opportunities', 'develop'],
        'Guidance & Support': ['support', 'mentoring', 'guidance', 'review', 'appraisal']
    },
    'Data & Workforce Analysis': {
        'Data Collection & Mapping': ['data', 'mapping', 'workforce', 'collect', 'analysis', 'understand'],
        'Surveys & Feedback': ['survey', 'surveys', 'feedback', 'pulse', 'culture'],
        'Skills Analysis': ['skills', 'skill', 'audit', 'needs', 'capability']
    },
    'EDI (Equality, Diversity & Inclusion)': {
        'EDI Initiatives & Strategy': ['edi', 'equality', 'diversity', 'inclusion', 'strategy', 'action', 'plan'],
        'Support & Working Groups': ['group', 'network', 'support', 'champions', 'working'],
        'Monitoring & Promotion': ['promote', 'monitor', 'ensure', 'fair', 'transparent', 'inclusive']
    },
    'External Collaboration & Partnerships': {
        'Engagement with External Partners': ['external', 'partners', 'collaborate', 'collaboration', 'stakeholders', 'industry'],
        'Regional & National Networks': ['networks', 'regional', 'national', 'uk', 'midlands', 'talent'],
        'Sharing Best Practice': ['sharing', 'practice', 'share', 'showcase', 'knowledge']
    },
    'Funding for Technicians': {
        'Securing Internal Funding': ['funding', 'fund', 'budget', 'resource', 'financial', 'support', 'central'],
        'External Funding Opportunities': ['external', 'grants', 'opportunities', 'bids', 'applications'],
        'Sustainability & Costing': ['sustainability', 'costing', 'recharging', 'sustainable', 'model']
    },
    'Mentorship & Support': {
        'Mentoring Schemes': ['mentoring', 'mentor', 'mentors', 'scheme', 'schemes', 'programme'],
        'Coaching & Guidance': ['coaching', 'coach', 'guidance', 'buddy', 'peer'],
        'Support Networks': ['support', 'network', 'networks', 'community', 'groups']
    },
    'Monitoring & Evaluation of TC': {
        'Progress Monitoring & Reporting': ['monitor', 'progress', 'evaluation', 'report', 'track', 'measure'],
        'Action Plan Management': ['action', 'plan', 'review', 'delivery', 'actions', 'objectives'],
        'Surveys & Metrics': ['survey', 'surveys', 'metrics', 'kpis', 'data', 'baseline']
    },
    'Networking and Presenting': {
        'Networking Events & Conferences': ['network', 'networking', 'events', 'conference', 'forum', 'symposium'],
        'Showcasing & Presenting': ['present', 'presenting', 'showcase', 'opportunities', 'posters', 'talks'],
        'Community Building': ['community', 'communities', 'practice', 'groups', 'connect']
    },
    'Ongoing Visibility & Communication': {
        'Internal Communications': ['communications', 'comms', 'newsletter', 'internal', 'channels', 'bulletin'],
        'Website & Online Profiles': ['website', 'online', 'profiles', 'web', 'webpages', 'digital'],
        'Promotion & Case Studies': ['promote', 'promotion', 'visibility', 'case', 'studies', 'highlight', 'showcase']
    },
    'Professional Registration & Accreditation': {
        'Promotion of Registration': ['professional', 'registration', 'promote', 'encourage', 'promotion'],
        'Support & Mentorship for Registration': ['support', 'mentors', 'champions', 'workshops', 'cohorts', 'guidance'],
        'Funding & Financial Support': ['funding', 'fund', 'fees', 'financial', 'costs', 'reimburse']
    },
    'Recognition & Awards': {
        'Awards & Recognition Schemes': ['awards', 'award', 'recognition', 'scheme', 'schemes', 'ceremony'],
        'Celebrating Success': ['celebrate', 'success', 'excellence', 'contribution', 'achievements'],
        'Promotion of Recognition': ['promote', 'raise', 'profile', 'ensure', 'recognise']
    },
    'Recruitment & Onboarding': {
        'Recruitment Processes': ['recruitment', 'recruiting', 'adverts', 'job', 'selection', 'hiring'],
        'Induction & Onboarding': ['induction', 'onboarding', 'welcome', 'new', 'starters'],
        'Improving Diversity in Recruitment': ['edi', 'diversity', 'inclusive', 'fair', 'transparent']
    },
    'Representation in Institutional Governance': {
        'Committee Representation': ['representation', 'committee', 'committees', 'groups', 'meetings'],
        'Voice & Influence': ['voice', 'involved', 'involvement', 'input', 'consultation'],
        'Policy & Decision Making': ['policy', 'decision', 'making', 'governance', 'strategy']
    },
    'Technician Leadership': {
        'Leadership Development': ['leadership', 'development', 'programme', 'skills', 'training'],
        'Management & Senior Roles': ['management', 'leaders', 'senior', 'managers', 'lead'],
        'Empowerment & Influence': ['empower', 'strategic', 'influence', 'voice', 'opportunities']
    },
    'Technician Voice & Feedback': {
        'Feedback Mechanisms': ['feedback', 'voice', 'forums', 'channels', 'mechanisms', 'listen'],
        'Surveys & Consultations': ['survey', 'surveys', 'consult', 'consultation', 'engagement'],
        'Representation & Advocacy': ['representation', 'representative', 'advocacy', 'champions', 'network']
    },
    'Training & Skills Development': {
        'Training Needs & Skills Audit': ['training', 'skills', 'development', 'needs', 'audit', 'analysis'],
        'Development Opportunities & Workshops': ['opportunities', 'workshops', 'courses', 'access', 'provide', 'sessions'],
        'Personal Development Plans (PDP)': ['pdp', 'pdps', 'personal', 'development', 'planning', 'review']
    }
}
# =====================================================================================


def assign_subcluster_name(statement, subcluster_definitions):
    """
    Assigns a statement to a predefined sub-cluster based on keyword matching.
    """
    # Clean the statement to ensure better matching
    statement_lower = statement.lower()
    
    # Use a dictionary to store the match count for each sub-cluster
    scores = {name: 0 for name in subcluster_definitions.keys()}
    
    # Tally the scores
    for name, keywords in subcluster_definitions.items():
        for keyword in keywords:
            # Use regex to find whole words to avoid matching parts of words (e.g., 'art' in 'department')
            if re.search(r'\b' + re.escape(keyword) + r'\b', statement_lower):
                scores[name] += 1
                
    # Find the sub-cluster with the highest score. If all scores are 0, it will return 'Unassigned'
    max_score = 0
    best_match = 'Unassigned'
    for name, score in scores.items():
        if score > max_score:
            max_score = score
            best_match = name
            
    return best_match


# --- Main Script Execution ---
try:
    df = pd.read_csv('Actions_FinalData - All_Data.csv')
    
    # Prepare a list to hold the results
    all_results = []
    
    print("Starting classification based on predefined sub-clusters...")
    
    # Loop through each unique category from the dataframe
    for category_name in df['Categories'].unique():
        print(f"Processing Category: '{category_name}'")
        
        # Filter the dataframe for the current category
        category_df = df[df['Categories'] == category_name].copy()
        
        # Check if we have defined sub-clusters for this category
        if category_name in category_to_subclusters:
            # Get the definitions for the current category
            subcluster_defs = category_to_subclusters[category_name]
            # Apply the classification function to each statement
            category_df['predefined_subcluster'] = category_df['Extracted Action'].apply(
                lambda x: assign_subcluster_name(x, subcluster_defs)
            )
        else:
            # If no definitions exist, mark all as 'Not Defined'
            category_df['predefined_subcluster'] = 'Not Defined'
            
        all_results.append(category_df)
        
    # Combine all processed dataframes back into one
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save the final results to a new CSV file
    output_filename = 'named_subcluster_analysis.csv'
    final_df.to_csv(output_filename, index=False)
    
    print("\nClassification complete!")
    print(f"All results have been saved to '{output_filename}'")

except FileNotFoundError:
    print("Error: The file 'Actions_FinalData - All_Data.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")