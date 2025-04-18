import React from 'react';
import Layout from '@theme/Layout';
import styles from './index.module.css';
import HomepageContent from '../components/HomepageContent';

export default function Home() {
  return (
    <Layout title="Welcome" description="Your site description here">
      <main className={styles.main}>
        <HomepageContent />
      </main>
    </Layout>
  );
}
