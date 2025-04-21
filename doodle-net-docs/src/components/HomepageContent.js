import React from 'react';
import Link from '@docusaurus/Link';
import { useColorMode } from '@docusaurus/theme-common';
import styles from './HomepageContent.module.css';

export default function HomepageContent() {
  const { colorMode } = useColorMode();
  const logoSrc = colorMode === 'dark' ? useBaseUrl('/img/doodlenet-dark.png') : useBaseUrl('/img/doodlenet-light.png');
  const dark_class = styles.link;

  return (
    <div className={styles.hero}>
      <img src={logoSrc} alt="Site Logo" className={styles.logo} />
      <Link className={`button button--primary button--lg ${dark_class}`} to="/docs/intro">
        Get Started
      </Link>
    </div>
  );
}
