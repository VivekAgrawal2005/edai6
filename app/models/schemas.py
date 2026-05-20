from typing import Optional

from pydantic import BaseModel


class EmailInput(BaseModel):

    email_id: Optional[str] = None

    sender: Optional[str] = None

    subject: Optional[str] = ""

    body: Optional[str] = ""

    timestamp: Optional[str] = None