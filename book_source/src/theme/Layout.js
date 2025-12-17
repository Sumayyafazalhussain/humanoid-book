import React from 'react';
import Layout from '@theme-original/Layout';
import ChatWidget from '@site/src/components/ChatBot';
import '@site/src/css/chatbot.css';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <ChatWidget />
    </>
  );
}
