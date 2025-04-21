from langchain_ollama import ChatOllama
import logging

class OllamaClientFactory:
    _clients = {}

    @classmethod
    def create_client(cls, role: str, model_name: str, base_url: str, temperature: float):
        """
        Yeni bir client oluşturur (varsa eskisini siler) ve singleton olarak kaydeder.
        """
        try:
            client = ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
            cls._clients[role] = client
            logging.info(f"OLLAMA client oluşturuldu/yenilendi: role={role}")
            return client
        except Exception as e:
            logging.error(f"{role} client oluşturulurken hata: {e}")
            raise RuntimeError(f"{role} client oluşturulurken hata: {e}")

    @classmethod
    def get_client(cls, role: str):
        """
        Daha önce create edilmiş client’ı döner. Yoksa hata fırlatır.
        """
        if role not in cls._clients:
            raise RuntimeError(f"{role} rolü için client bulunamadı! Önce create_client çağırın.")
        return cls._clients[role]

    @classmethod
    def delete_client(cls, role: str):
        """
        Belirli bir role ait instance’ı kaldırır.
        """
        if role in cls._clients:
            del cls._clients[role]
            logging.info(f"OLLAMA client silindi: role={role}")

    @classmethod
    def list_roles(cls):
        """
        Şu anda hangi roller için client var, listeler.
        """
        return list(cls._clients.keys())
