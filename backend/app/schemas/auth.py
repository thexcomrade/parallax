from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(
        min_length=12, max_length=128, description="Password must be 12-128 characters"
    )


class UserRoleUpdate(BaseModel):
    role: str = Field(
        ...,
        description="User role: 'public', 'operator', or 'admin'",
        pattern="^(public|operator|admin)$",
    )


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    role: str = "public"
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageResponse(BaseModel):
    message: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(
        min_length=12, max_length=128, description="Password must be 12-128 characters"
    )
