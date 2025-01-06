from dataclasses import dataclass

@dataclass
class DirectorySetting:
    """set the directory for the model download"""
    home_dir: str=""
    transformers_cache_home: str=""
    huggingface_token_file: str=""

    def get_cache_home(self):
        """get the cache home"""
        return f"{self.home_dir}\\{self.transformers_cache_home}"
    
    def get_token_file(self):
        """get the token file"""
        return f"{self.home_dir}\\{self.huggingface_token_file}"
    
DIR_MODE_MAP = {
    "win_local": DirectorySetting(home_dir="C:\\Users\\yingdingwang",
                                transformers_cache_home="MODELS", 
                                huggingface_token_file="MODELS\\.huggingface_token")
}