# DECISION TREE - CHURN PREDICTION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# DATA PIPELINE
class ChurnPredictor:
    """Efficient Churn Prediction Pipeline"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.selected_features = None
        self.le = LabelEncoder()
        
    def load_and_clean_data(self):
        """Combined loading and cleaning for efficiency"""
        print("="*60)
        print("📊 LOADING & CLEANING DATA")
        print("="*60)
        
        try:
            df = pd.read_csv(self.file_path)
            print(f"✅ Dataset loaded: {df.shape}")
        except FileNotFoundError:
            print(f"❌ File not found: {self.file_path}")
            return None, None
        
        if 'Churn' in df.columns:
            churn_counts = df['Churn'].value_counts()
            print(f"📊 Churn Distribution: Stay={churn_counts.get(0,0)} ({churn_counts.get(0,0)/len(df)*100:.1f}%), "
                  f"Churn={churn_counts.get(1,0)} ({churn_counts.get(1,0)/len(df)*100:.1f}%)")
        
        # Cleaning remove unnecessary columns and data leakage features
        drop_columns = ['customerID', 'Unnamed: 0', 'PromptInput', 'CustomerFeedback', 'sentiment', 'feedback_length']
        df_clean = df.drop(columns=[col for col in drop_columns if col in df.columns])
        
        removed = [col for col in drop_columns if col in df.columns]
        print(f"🗑️  Removed columns: {removed}")
        
        X = df_clean.drop('Churn', axis=1)
        y = df_clean['Churn']
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = self.le.fit_transform(X[col])
        
        print(f"✅ Encoded {len(categorical_cols)} categorical columns")
        print(f"✅ Final features: {list(X.columns)}")
        
        return X, y
    
    def select_features(self, X, y, k=8):
        """Efficient feature selection"""
        print(f"\n🎯 SELECTING TOP {k} FEATURES")
        print("-" * 40)
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]
        
        # Display selected features
        for i, (feature, score) in enumerate(zip(self.selected_features, feature_scores), 1):
            print(f"  {i:2d}. {feature:<20} ({score:.4f})")
        
        return pd.DataFrame(X_selected, columns=self.selected_features)
    
    def optimize_and_train(self, X_train, y_train):
        """Efficient hyperparameter tuning and training"""
        print(f"\n⚙️  OPTIMIZING HYPERPARAMETERS")
        print("-" * 40)
        
        # Streamlined parameter grid (most effective combinations only)
        param_grid = {
            'max_depth': [2, 3],
            'min_samples_split': [500, 1000],
            'min_samples_leaf': [200, 300],
            'criterion': ['gini']  # Use Gini typically faster than entropy
        }
        
        # Efficient grid search with reduced CV folds
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42, max_features='sqrt'),
            param_grid, 
            cv=3,
            scoring='f1',
            n_jobs=-1  # Use all CPU cores
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"✅ Best params: {grid_search.best_params_}")
        print(f"✅ Best F1 score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test, show_plot=True):
        """Comprehensive but efficient evaluation"""
        print(f"\n📊 MODEL EVALUATION")
        print("-" * 40)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Core metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        gap = train_acc - test_acc
        
        # Display results
        print(f"📈 Training Accuracy: {train_acc:.4f}")
        print(f"📈 Testing Accuracy:  {test_acc:.4f}")
        print(f"📈 ROC-AUC Score:     {roc_auc:.4f}")
        print(f"📈 Train-Test Gap:    {gap:.4f}")
        
        # Validation check
        issues = []
        if test_acc > 0.90: issues.append("Test accuracy too high")
        if roc_auc > 0.95: issues.append("ROC-AUC too high") 
        if gap > 0.10: issues.append("Large train-test gap")
        if train_acc > 0.95: issues.append("Training accuracy too high")
        
        if issues:
            print(f"⚠️  Issues detected: {', '.join(issues)}")
            is_valid = False
        else:
            print("✅ Model validation passed!")
            is_valid = True
        
        # Classification report (compact format)
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Stay', 'Churn']))
        
        # Optional confusion matrix plot
        if show_plot:
            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
        
        return train_acc, test_acc, roc_auc, is_valid
    
    def cross_validate(self, X, y):
        """Quick cross-validation"""
        print(f"\n✅ CROSS-VALIDATION")
        print("-" * 40)
        
        cv_acc = cross_val_score(self.model, X, y, cv=3, scoring='accuracy')
        cv_f1 = cross_val_score(self.model, X, y, cv=3, scoring='f1')
        
        print(f"📊 CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"📊 CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        return cv_acc.mean() <= 0.90  # Return if acceptable
    
    def run_full_pipeline(self, test_size=0.2, n_features=8, show_plots=True):
        """Complete pipeline in one method"""
        print("🚀 STARTING EFFICIENT CHURN PREDICTION PIPELINE")
        print("=" * 60)
        
        # Step 1 - Load and clean
        X, y = self.load_and_clean_data()
        if X is None:
            return None
        
        # Step 2 - Feature selection
        X_selected = self.select_features(X, y, k=n_features)
        
        # Step 3 - Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"\n📊 Data split: Train={X_train.shape}, Test={X_test.shape}")
        
        # Step 4 - Optimize and train
        self.optimize_and_train(X_train, y_train)
        
        # Step 5 - Evaluate
        train_acc, test_acc, roc_auc, is_valid = self.evaluate_model(
            X_train, X_test, y_train, y_test, show_plots
        )
        
        # Step 6 - Cross-validate
        cv_valid = self.cross_validate(X_selected, y)
        
        # Final summary
        self.print_final_summary(train_acc, test_acc, roc_auc, is_valid and cv_valid)
        
        return {
            'model': self.model,
            'features': self.selected_features,
            'performance': {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'roc_auc': roc_auc,
                'is_production_ready': is_valid and cv_valid
            }
        }
    
    def print_final_summary(self, train_acc, test_acc, roc_auc, is_production_ready):
        """Compact final summary"""
        print(f"\n" + "=" * 60)
        print("🎯 FINAL RESULTS")
        print("=" * 60)
        
        print(f"🔧 Model Config: Depth={self.model.max_depth}, "
              f"MinSplit={self.model.min_samples_split}, "
              f"MinLeaf={self.model.min_samples_leaf}")
        
        print(f"📊 Performance: Accuracy={test_acc:.3f}, "
              f"ROC-AUC={roc_auc:.3f}, "
              f"Gap={train_acc-test_acc:.3f}")
        
        status = "✅ PRODUCTION READY" if is_production_ready else "❌ NEEDS IMPROVEMENT"
        print(f"🎯 Status: {status}")
        print("=" * 60)

# EXECUTION
if __name__ == "__main__":
    # File path
    file_path = "C:/Users/lenovo/Documents/Semester 4/Kecerdasan Buatan/Project AI UAS/telco_prep.csv"
    
    # Create and run pipeline
    predictor = ChurnPredictor(file_path)
    
    # Run complete pipeline with efficient settings
    results = predictor.run_full_pipeline(
        test_size=0.2,
        n_features=8,
        show_plots=True  # Set False untuk lebih cepat
    )
    
    """
    if results and results['performance']['is_production_ready']:
        print("\n🔄 Creating balanced version for better churn detection...")
        
        balanced_model = DecisionTreeClassifier(
            **predictor.model.get_params(),
            class_weight='balanced'
        )
        
        # Quick refit and test
        balanced_model.fit(X_train, y_train)
        balanced_pred = balanced_model.predict(X_test)
        
        print("📊 Balanced Model Results:")
        print(classification_report(y_test, balanced_pred, target_names=['Stay', 'Churn']))
    """
    
    print("\n🎉 EFFICIENT PIPELINE COMPLETE!")