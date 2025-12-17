const API_URL = 'http://localhost:8000'; // Ensure this matches your backend's URL

export const query = async (userQuery) => {
  const response = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: userQuery }),
  });
  if (!response.ok) {
    // Attempt to parse error response from backend if available
    const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new Error(errorData.message || 'Network response was not ok');
  }
  return response.json();
};

export const querySelectedText = async (userQuery, selectedText) => {
  const response = await fetch(`${API_URL}/query_selected_text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: userQuery, selected_text: selectedText }),
  });
  if (!response.ok) {
    // Attempt to parse error response from backend if available
    const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new Error(errorData.message || 'Network response was not ok');
  }
  return response.json();
};
