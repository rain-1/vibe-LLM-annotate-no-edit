# API Documentation

## Overview

This API provides endpoints for managing user accounts and authentication. All requests must include a valid API key in the header.

## Authentication

### POST /auth/login

Authenticates a user and returns a JWT token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secretpassword"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

### POST /auth/logout

Invalidates the current session token.

**Headers:**
- `Authorization: Bearer <token>`

## User Management

### GET /users/{id}

Retrieves user information by ID.

**Parameters:**
- `id` (path): The user's unique identifier

**Response:**
```json
{
  "id": "123",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### PUT /users/{id}

Updates user information.

**Request Body:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request |
| 401  | Unauthorized |
| 404  | Not Found |
| 500  | Internal Server Error |
